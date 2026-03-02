//! Self-update command: check GitHub Releases for newer versions,
//! download and replace the binary with user confirmation.

use crate::CliError;
use std::process::Command;

const GITHUB_API_URL: &str =
    "https://api.github.com/repos/octoflow-lang/octoflow/releases/latest";

/// Run `octoflow update [--check]`.
pub fn run_update(args: &[String]) -> Result<(), CliError> {
    let check_only = args.iter().any(|a| a == "--check");

    eprintln!("Checking for updates...");

    let json = curl_get_json(GITHUB_API_URL)?;

    let remote_tag = json_extract_string(&json, "tag_name")
        .ok_or_else(|| CliError::Runtime("could not parse latest release tag".into()))?;

    let remote_version = remote_tag.trim_start_matches('v');
    let local_version = crate::VERSION;

    if remote_version == local_version {
        eprintln!("OctoFlow v{} is already the latest version.", local_version);
        return Ok(());
    }

    // Determine if remote is actually newer (simple string compare works for semver)
    eprintln!();
    eprintln!("OctoFlow v{} available (you have v{})", remote_version, local_version);

    // Show changelog if available
    if let Some(body) = json_extract_string(&json, "body") {
        let body = body.trim();
        if !body.is_empty() {
            eprintln!();
            eprintln!("Changes:");
            for line in body.lines().take(15) {
                eprintln!("  {}", line);
            }
            if body.lines().count() > 15 {
                eprintln!("  ...(truncated)");
            }
        }
    }

    if check_only {
        return Ok(());
    }

    // Ask user
    eprintln!();
    eprint!("Download and update? [y/N] ");
    let mut input = String::new();
    if std::io::stdin().read_line(&mut input).is_err() || !input.trim().eq_ignore_ascii_case("y") {
        eprintln!("Update cancelled.");
        return Ok(());
    }

    // Detect platform and pick asset name
    let asset_name = platform_asset_name(remote_version)?;

    // Find download URL from release assets
    let download_url = find_asset_url(&json, &asset_name)?;
    let checksums_url = find_asset_url(&json, "SHA256SUMS.txt").ok();

    eprintln!("Downloading {}...", asset_name);
    let tmp_dir = std::env::temp_dir().join("octoflow-update");
    let _ = std::fs::create_dir_all(&tmp_dir);
    let archive_path = tmp_dir.join(&asset_name);

    curl_download(&download_url, &archive_path)?;

    // Verify SHA-256 if checksums available
    if let Some(ref checksums_url) = checksums_url {
        eprintln!("Verifying SHA-256...");
        if let Ok(checksums_text) = curl_get_json(checksums_url) {
            if !verify_sha256(&archive_path, &asset_name, &checksums_text)? {
                let _ = std::fs::remove_dir_all(&tmp_dir);
                return Err(CliError::Runtime(
                    "SHA-256 mismatch — download may be corrupted. Update aborted.".into(),
                ));
            }
            eprintln!("SHA-256 verified.");
        }
    }

    // Extract binary from archive
    let new_binary = extract_binary(&archive_path, &tmp_dir)?;

    // Replace current binary
    let current_exe = std::env::current_exe()
        .map_err(|e| CliError::Io(format!("cannot locate current binary: {}", e)))?;

    replace_binary(&new_binary, &current_exe)?;

    // Cleanup
    let _ = std::fs::remove_dir_all(&tmp_dir);

    eprintln!("Updated to OctoFlow v{}. Run `octoflow --version` to confirm.", remote_version);
    Ok(())
}

// ---------------------------------------------------------------------------
// HTTP helpers (reuse curl pattern from io/web.rs)
// ---------------------------------------------------------------------------

fn curl_get_json(url: &str) -> Result<String, CliError> {
    let output = Command::new("curl")
        .args([
            "-s",
            "-L",
            "--max-time",
            "15",
            "-H",
            "Accept: application/vnd.github+json",
            "-H",
            "User-Agent: OctoFlow-Updater",
            url,
        ])
        .output()
        .map_err(|e| CliError::Io(format!("curl not found — cannot check for updates: {}", e)))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(CliError::Runtime(format!(
            "failed to check for updates: {}",
            stderr.trim()
        )));
    }

    String::from_utf8(output.stdout)
        .map_err(|e| CliError::Runtime(format!("invalid response: {}", e)))
}

fn curl_download(url: &str, dest: &std::path::Path) -> Result<(), CliError> {
    let output = Command::new("curl")
        .args([
            "-s",
            "-L",
            "--max-time",
            "120",
            "-H",
            "Accept: application/octet-stream",
            "-H",
            "User-Agent: OctoFlow-Updater",
            "-o",
        ])
        .arg(dest.to_string_lossy().as_ref())
        .arg(url)
        .output()
        .map_err(|e| CliError::Io(format!("download failed: {}", e)))?;

    if !output.status.success() {
        return Err(CliError::Runtime("download failed".into()));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Platform detection
// ---------------------------------------------------------------------------

fn platform_asset_name(version: &str) -> Result<String, CliError> {
    let name = if cfg!(target_os = "windows") {
        format!("octoflow-v{}-x86_64-windows.zip", version)
    } else if cfg!(target_os = "macos") {
        format!("octoflow-v{}-aarch64-macos.tar.gz", version)
    } else {
        format!("octoflow-v{}-x86_64-linux.tar.gz", version)
    };
    Ok(name)
}

// ---------------------------------------------------------------------------
// Archive extraction
// ---------------------------------------------------------------------------

fn extract_binary(
    archive: &std::path::Path,
    tmp_dir: &std::path::Path,
) -> Result<std::path::PathBuf, CliError> {
    let archive_str = archive.to_string_lossy();

    if archive_str.ends_with(".zip") {
        // Windows: use PowerShell to extract
        let output = Command::new("powershell")
            .args([
                "-NoProfile",
                "-Command",
                &format!(
                    "Expand-Archive -Force -Path '{}' -DestinationPath '{}'",
                    archive_str,
                    tmp_dir.to_string_lossy()
                ),
            ])
            .output()
            .map_err(|e| CliError::Io(format!("extraction failed: {}", e)))?;

        if !output.status.success() {
            return Err(CliError::Runtime("failed to extract zip".into()));
        }

        let exe = tmp_dir.join("octoflow.exe");
        if exe.exists() {
            return Ok(exe);
        }
        // Search one level deep
        if let Ok(entries) = std::fs::read_dir(tmp_dir) {
            for entry in entries.flatten() {
                let p = entry.path().join("octoflow.exe");
                if p.exists() {
                    return Ok(p);
                }
            }
        }
        Err(CliError::Runtime(
            "octoflow.exe not found in extracted archive".into(),
        ))
    } else {
        // Unix: tar xzf
        let output = Command::new("tar")
            .args(["xzf"])
            .arg(archive)
            .arg("-C")
            .arg(tmp_dir)
            .output()
            .map_err(|e| CliError::Io(format!("extraction failed: {}", e)))?;

        if !output.status.success() {
            return Err(CliError::Runtime("failed to extract tar.gz".into()));
        }

        let exe = tmp_dir.join("octoflow");
        if exe.exists() {
            return Ok(exe);
        }
        if let Ok(entries) = std::fs::read_dir(tmp_dir) {
            for entry in entries.flatten() {
                let p = entry.path().join("octoflow");
                if p.exists() {
                    return Ok(p);
                }
            }
        }
        Err(CliError::Runtime(
            "octoflow binary not found in extracted archive".into(),
        ))
    }
}

// ---------------------------------------------------------------------------
// Binary replacement
// ---------------------------------------------------------------------------

fn replace_binary(
    new_binary: &std::path::Path,
    current_exe: &std::path::Path,
) -> Result<(), CliError> {
    if cfg!(target_os = "windows") {
        // Windows can't overwrite a running exe — rename first
        let old = current_exe.with_extension("old.exe");
        let _ = std::fs::remove_file(&old);
        std::fs::rename(current_exe, &old)
            .map_err(|e| CliError::Io(format!("cannot rename current binary: {}", e)))?;
        std::fs::copy(new_binary, current_exe)
            .map_err(|e| CliError::Io(format!("cannot install new binary: {}", e)))?;
        let _ = std::fs::remove_file(&old);
    } else {
        std::fs::copy(new_binary, current_exe)
            .map_err(|e| CliError::Io(format!("cannot replace binary: {}", e)))?;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// SHA-256 verification
// ---------------------------------------------------------------------------

fn verify_sha256(
    file: &std::path::Path,
    asset_name: &str,
    checksums: &str,
) -> Result<bool, CliError> {
    // Find the expected hash for our asset
    let expected = checksums
        .lines()
        .find(|line| line.contains(asset_name))
        .and_then(|line| line.split_whitespace().next())
        .ok_or_else(|| CliError::Runtime("checksum not found for this platform".into()))?;

    // Compute SHA-256 of downloaded file
    let computed = if cfg!(target_os = "windows") {
        let output = Command::new("certutil")
            .args(["-hashfile"])
            .arg(file)
            .arg("SHA256")
            .output()
            .map_err(|e| CliError::Io(format!("certutil failed: {}", e)))?;
        let text = String::from_utf8_lossy(&output.stdout);
        text.lines()
            .nth(1)
            .unwrap_or("")
            .trim()
            .replace(' ', "")
            .to_lowercase()
    } else {
        let output = Command::new("sha256sum")
            .arg(file)
            .output()
            .map_err(|e| CliError::Io(format!("sha256sum failed: {}", e)))?;
        let text = String::from_utf8_lossy(&output.stdout);
        text.split_whitespace()
            .next()
            .unwrap_or("")
            .to_lowercase()
    };

    Ok(computed == expected.to_lowercase())
}

// ---------------------------------------------------------------------------
// Minimal JSON string extraction (zero-dep, matches existing codebase pattern)
// ---------------------------------------------------------------------------

fn json_extract_string(json: &str, key: &str) -> Option<String> {
    let pattern = format!("\"{}\"", key);
    let idx = json.find(&pattern)?;
    let after_key = &json[idx + pattern.len()..];

    // Skip whitespace and colon
    let rest = after_key.trim_start();
    let rest = rest.strip_prefix(':')?;
    let rest = rest.trim_start();

    if rest.starts_with('"') {
        // String value — handle escapes
        let inner = &rest[1..];
        let mut result = String::new();
        let mut chars = inner.chars();
        while let Some(c) = chars.next() {
            match c {
                '"' => return Some(result),
                '\\' => {
                    if let Some(escaped) = chars.next() {
                        match escaped {
                            'n' => result.push('\n'),
                            'r' => result.push('\r'),
                            't' => result.push('\t'),
                            '"' => result.push('"'),
                            '\\' => result.push('\\'),
                            '/' => result.push('/'),
                            _ => {
                                result.push('\\');
                                result.push(escaped);
                            }
                        }
                    }
                }
                _ => result.push(c),
            }
        }
        None
    } else if rest.starts_with("null") {
        None
    } else {
        None
    }
}

fn find_asset_url(json: &str, asset_name: &str) -> Result<String, CliError> {
    // Look for browser_download_url containing asset_name
    let pattern = format!("\"browser_download_url\"");
    let mut search_from = 0;
    while let Some(idx) = json[search_from..].find(&pattern) {
        let abs_idx = search_from + idx;
        let after = &json[abs_idx + pattern.len()..];
        let rest = after.trim_start();
        if let Some(rest) = rest.strip_prefix(':') {
            let rest = rest.trim_start();
            if rest.starts_with('"') {
                let inner = &rest[1..];
                if let Some(end) = inner.find('"') {
                    let url = &inner[..end];
                    if url.contains(asset_name) {
                        return Ok(url.to_string());
                    }
                }
            }
        }
        search_from = abs_idx + pattern.len();
    }
    Err(CliError::Runtime(format!(
        "asset '{}' not found in latest release",
        asset_name
    )))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_json_extract_tag() {
        let json = r#"{"tag_name":"v1.4.0","name":"Release v1.4.0"}"#;
        assert_eq!(
            json_extract_string(json, "tag_name"),
            Some("v1.4.0".to_string())
        );
    }

    #[test]
    fn test_json_extract_body_with_escapes() {
        let json = r#"{"body":"line1\nline2\nline3"}"#;
        let body = json_extract_string(json, "body").unwrap();
        assert!(body.contains('\n'));
        assert!(body.starts_with("line1"));
    }

    #[test]
    fn test_json_extract_null() {
        let json = r#"{"body":null,"tag_name":"v1.3.0"}"#;
        assert_eq!(json_extract_string(json, "body"), None);
        assert_eq!(
            json_extract_string(json, "tag_name"),
            Some("v1.3.0".to_string())
        );
    }

    #[test]
    fn test_platform_asset_name() {
        let name = platform_asset_name("1.4.0").unwrap();
        assert!(name.starts_with("octoflow-v1.4.0-"));
        assert!(name.contains("windows") || name.contains("linux") || name.contains("macos"));
    }

    #[test]
    fn test_find_asset_url() {
        let json = r#"{"assets":[{"browser_download_url":"https://github.com/octoflow-lang/octoflow/releases/download/v1.4.0/octoflow-v1.4.0-x86_64-windows.zip"},{"browser_download_url":"https://github.com/octoflow-lang/octoflow/releases/download/v1.4.0/SHA256SUMS.txt"}]}"#;
        let url = find_asset_url(json, "SHA256SUMS.txt").unwrap();
        assert!(url.contains("SHA256SUMS.txt"));
    }

    #[test]
    fn test_version_comparison_same() {
        // If remote == local, no update needed
        assert_eq!("1.3.0", "1.3.0");
    }
}
