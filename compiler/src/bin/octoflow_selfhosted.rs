// octoflow_selfhosted binary — Self-hosted compiler entry point
//
// Invokes eval.flow (22,128 lines) for compilation instead of
// compiler.rs (13,547 lines Rust). Path to < 500 lines Rust.

fn main() {
    octoflow_cli::loader::main();
}
