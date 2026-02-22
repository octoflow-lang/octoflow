# Changelog

## [1.0.0] - 2026-02-23

### Added
- GPU VM: general-purpose virtual machine with 5-SSBO memory model
- GPU VM: register-based inter-instance message passing (R30→R31)
- GPU VM: homeostasis regulator (activation stability, throughput tracking)
- GPU VM: indirect dispatch — GPU self-programs workgroup counts
- GPU VM: CPU polling with HOST_VISIBLE Metrics/Control for zero-copy reads
- GPU VM: dormant VM activation without command buffer rebuild
- GPU VM: I/O streaming — CPU streams data to Globals, GPU processes without restart
- DB engine: columnar storage in GPU memory with fused query kernels
- DB compression: delta encoding, dictionary lookup, Q4_K dequantization
- DB compression: compressed query chains (decompress → WHERE → aggregate), bit-exact
- 51 stdlib modules (collections, data, db, ml, science, stats, string, sys, time, web, core)
- Security sandbox with --allow-read, --allow-write, --allow-net, --allow-exec
- REPL with GPU support
- 658 tests passing (648 Rust + 10 .flow integration)

### Fixed
- Uninitialized vector UB in device.rs (P0)
- Missing read permission checks in load_file() (P1)
- 12 manual div_ceil reimplementations replaced with .div_ceil() (P1)
- Deduplicated video_open/video_frame handlers (-172 lines) (P2)

## [0.83.1] - 2026-02-19
- Initial public release
- GPU-native language with Vulkan compute backend
- 51 stdlib modules
- Windows x64 and Linux x64 binaries
