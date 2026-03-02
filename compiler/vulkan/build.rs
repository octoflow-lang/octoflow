// Build script for octoflow-vulkan — adds Vulkan SDK lib path.
// Windows: links vulkan-1.lib from Vulkan SDK or system path.
// Linux:   links libvulkan.so from SDK, /usr/lib, or pkg-config.
// macOS:   links libvulkan.dylib from MoltenVK/SDK or Homebrew.

fn main() {
    // VULKAN_SDK env var — set by official SDK installers on all platforms
    if let Ok(sdk) = std::env::var("VULKAN_SDK") {
        if cfg!(target_os = "windows") {
            println!("cargo:rustc-link-search={}/Lib", sdk);
        } else {
            println!("cargo:rustc-link-search={}/lib", sdk);
        }
    }

    #[cfg(target_os = "windows")]
    {
        // Fallback: scan known Windows install paths (newest first)
        let candidates = scan_vulkan_sdk_windows();
        for path in &candidates {
            if std::path::Path::new(path).exists() {
                println!("cargo:rustc-link-search={}", path);
                break;
            }
        }
        println!("cargo:rustc-link-lib=vulkan-1");
    }

    #[cfg(target_os = "linux")]
    {
        // Fallback: common Linux paths where libvulkan.so lives
        for candidate in &[
            "/usr/lib/x86_64-linux-gnu",    // Debian/Ubuntu
            "/usr/lib64",                    // Fedora/RHEL
            "/usr/lib",                      // Arch
        ] {
            if std::path::Path::new(&format!("{}/libvulkan.so", candidate)).exists() {
                println!("cargo:rustc-link-search={}", candidate);
                break;
            }
        }
        println!("cargo:rustc-link-lib=vulkan");
    }

    #[cfg(target_os = "macos")]
    {
        // MoltenVK via Homebrew or Vulkan SDK
        for candidate in &[
            "/usr/local/lib",                           // Homebrew Intel
            "/opt/homebrew/lib",                        // Homebrew Apple Silicon
            "/usr/local/share/vulkan/icd.d/../../../lib", // MoltenVK
        ] {
            if std::path::Path::new(&format!("{}/libvulkan.dylib", candidate)).exists()
                || std::path::Path::new(&format!("{}/libMoltenVK.dylib", candidate)).exists()
            {
                println!("cargo:rustc-link-search={}", candidate);
                break;
            }
        }
        println!("cargo:rustc-link-lib=vulkan");
    }
}

/// Scan C:\VulkanSDK\ for installed versions, return Lib paths newest-first.
#[cfg(target_os = "windows")]
fn scan_vulkan_sdk_windows() -> Vec<String> {
    let base = r"C:\VulkanSDK";
    let mut paths = Vec::new();
    if let Ok(entries) = std::fs::read_dir(base) {
        for entry in entries.flatten() {
            let lib = entry.path().join("Lib");
            if lib.exists() {
                paths.push(lib.to_string_lossy().into_owned());
            }
        }
    }
    // Sort descending so newest SDK version is tried first
    paths.sort_unstable_by(|a, b| b.cmp(a));
    paths
}
