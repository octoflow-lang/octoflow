# Loom Engine (L1)
GPU compute runtime — Vulkan dispatch, JIT kernels, NN layers, data ops.

## Architecture
Loom Engine = Main Loom + Support Loom. Neither exists alone.
- **Main Loom**: GPU-only compute. 1-way receive (dispatches). N per app.
- **Support Loom**: CPU↔GPU I/O bridge. Boot, upload, download, present. 1 per app.
- Three patterns: Implicit (CPU as Support), Conflated (single VM), Explicit (2 VMs).
- Multi-threading: Support Loom manages CPU threads transparently. No thread code in .flow.
- Homeostasis: auto-paces dispatches via timing (EMA baseline, 20% threshold).

## Key Functions
loom_boot loom_dispatch loom_dispatch_jit loom_build loom_run loom_launch
loom_poll loom_wait loom_read loom_write loom_present loom_shutdown loom_free
loom_set_heap loom_status loom_pace loom_prefetch

## Modules
runtime, ops, patterns, debug, nn/*, math/*, data/*

## See: L2-loom-*.md
