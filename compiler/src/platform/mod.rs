//! Platform-specific modules — OS bindings, GPU monitoring, windowing, text rendering.

#[cfg(target_os = "windows")]
pub mod win32;
#[cfg(target_os = "windows")]
pub mod text_render;
#[cfg(target_os = "windows")]
pub mod nvml;
#[cfg(target_os = "windows")]
pub mod cpu;
pub mod window;
pub mod audio;
