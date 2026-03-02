//! Token-by-token streaming output with ANSI formatting.
//!
//! Provides colored terminal output for the OctoFlow chat interface:
//! - Streaming tokens in cyan
//! - Error messages in red
//! - Retry/fix messages in yellow
//! - Success messages in green
//! - Spinner during model loading

/// ANSI color codes for terminal output.
pub const RESET: &str = "\x1b[0m";
pub const BOLD: &str = "\x1b[1m";
pub const CYAN: &str = "\x1b[36m";
pub const RED: &str = "\x1b[31m";
pub const YELLOW: &str = "\x1b[33m";
pub const GREEN: &str = "\x1b[32m";
pub const DIM: &str = "\x1b[2m";

/// Print the chat banner with ANSI colors.
pub fn print_banner() {
    eprintln!("{}{}OctoFlow Chat v{}{}", BOLD, CYAN, crate::VERSION, RESET);
    eprintln!("{}Type a description, get working .flow code. :help for commands.{}", DIM, RESET);
    eprintln!();
}

/// Print a colored status message.
pub fn print_status(msg: &str, color: &str) {
    eprint!("{}{}{}", color, msg, RESET);
}

/// Print an error message in red.
pub fn print_error(msg: &str) {
    eprintln!("{}error: {}{}", RED, msg, RESET);
}

/// Print a success message in green.
pub fn print_success(msg: &str) {
    eprintln!("{}✓ {}{}", GREEN, msg, RESET);
}

/// Print a retry/fix message in yellow.
pub fn print_retry(attempt: usize, max: usize) {
    eprintln!("{}Fixing... ({}/{}){}", YELLOW, attempt, max, RESET);
}

/// Clear the current line (for spinner removal).
pub fn clear_line() {
    eprint!("\r\x1b[K");
}
