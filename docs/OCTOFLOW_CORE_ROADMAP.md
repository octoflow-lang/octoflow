# OctoFlow Core Roadmap

## The Final Three Before Convergence

*Ship these, then return to F-DHGNN / OctoBrain*

---

## Current State

OctoFlow has proven its foundation across 116+ phases:

- 9 domains proven (database, ML, crypto, graph, scientific, compression, game logic, ETL, terminal compute)
- 41 GPU kernels, 32 verified stdlib functions, 5 BIT EXACT keystone primitives
- GPU-resident autonomous agents at 99.9999% GPU autonomy
- Self-hosted compiler (eval.flow) at 89/92 test parity
- Zero Rust changes across the entire GPU runtime track
- OpAtomicIAdd in the SPIR-V emitter

Three core features remain before public release and the return to convergence work (F-DHGNN + OctoBrain). Each builds on OctoFlow's existing architecture. Each is unique — no other GPU framework has them. Together, they complete OctoFlow as a self-sufficient GPU-native platform.

---

## Pre-Release Prerequisites

Before the three core features, these must close:

| Task | Status | Remaining |
|------|--------|-----------|
| eval.flow test parity | 89/92 (96.7%) | Fix conditional return + gcd timeout |
| Rust deletion | 18,000 lines redundant | Delete after 92/92 parity |
| Decoupled lookback scan | Blocked on ir.flow atomics | Single-pass scan with OpAtomicIAdd |

Once complete: OctoFlow is self-hosted with <500 lines of Rust (OS boundary loader only), and the keystone scan primitive runs at maximum performance.

---

## Core Feature 1: Homeostasis

### The GPU's Safety Valve

**What it is:** Runtime-level GPU self-regulation. The GPU monitors its own thermal state, power draw, and utilization, then automatically paces dispatch chain submission to maintain sustainable operation.

**Why it matters:** Every existing GPU framework treats hardware as disposable — 100% utilization until thermal throttle or failure. For 24/7 workloads (trading systems, streaming analytics, edge deployment, autonomous agents), this is destructive. Thermal spikes cause driver-forced downclocks that crater throughput. Power spikes increase operating costs. Sustained maximum utilization accelerates silicon degradation.

Homeostasis makes OctoFlow the first GPU-native language that cares about the hardware it runs on.

**The problem it solves:**

```
Without homeostasis:
  100% → 100% → 100% → THERMAL THROTTLE → 60% → 100% → 100% → THROTTLE → 60%
  Average sustained throughput: ~85%, with unpredictable latency spikes

With homeostasis:
  92% → 92% → 92% → 92% → 92% → 92% → 92% → 92% → 92%
  Average sustained throughput: ~92%, predictable and stable
```

Steady 92% beats spiking between 100% and 60%. Every time.

### Design

Four self-regulation mechanisms:

**Thermal Awareness**
- Query GPU temperature via Vulkan physical device properties / platform APIs
- Define a thermal target (e.g., 75°C) below the driver's throttle threshold
- As temperature approaches target, the runtime increases inter-chain spacing
- Result: the GPU never hits the thermal cliff where the driver forces massive downclocking

**Power Budgeting**
- User sets a power envelope (e.g., 150W)
- Runtime estimates power draw from dispatch density × kernel complexity × clock speed
- Adjusts dispatch pacing to stay within budget
- Use case: trading systems that don't need maximum throughput during off-hours, VPS instances with power constraints, battery-powered edge devices

**Wear Leveling**
- Distribute work across compute units rather than saturating the same ones
- Vary workgroup dispatch patterns to spread thermal load across the die
- Long-term: extends GPU hardware lifespan for always-on deployments

**Adaptive Duty Cycling**
- For continuous workloads (streaming, agents, monitoring), insert calculated idle gaps between dispatch chains
- The GPU breathes — thermals stabilize, sustained throughput increases
- Duty cycle is configurable: 85% for balanced, 95% for performance-critical, 70% for energy-saving

### API Surface

```
// Enable self-regulation
rt_homeostasis_enable()

// Configure thresholds
rt_set_thermal_target(75.0)       // degrees Celsius
rt_set_power_budget(150.0)        // watts
rt_set_duty_cycle(0.85)           // utilization ceiling (0.0 - 1.0)

// Existing dispatch chains work unchanged
rt_chain_begin()
  // ... dispatches ...
rt_chain_end()
// Runtime automatically paces submission based on homeostasis state

// Query current state
let temp = rt_gpu_temperature()
let power = rt_gpu_power_draw()
let util = rt_gpu_utilization()

// Profiles for common scenarios
rt_homeostasis_profile("sustained")    // 24/7 workloads
rt_homeostasis_profile("burst")        // short high-intensity tasks
rt_homeostasis_profile("quiet")        // background / idle processing
```

### Implementation Scope

- Temperature query: FFI to Vulkan device properties or platform-specific API (NVML, AMD SMI)
- Pacing logic: `.flow` wrapper around `rt_chain_submit` with adaptive timing
- Power estimation: derived from dispatch metrics (approximation sufficient for v1)
- Duty cycle: timer-based submission gating between chain completions
- Estimated size: ~500 lines of `.flow` + 2-3 FFI functions for hardware telemetry
- No kernel changes. No runtime architecture changes. Wraps existing submission path.

### Why It's First

Smallest build of the three core features. Highest novelty (nobody has this). Self-contained — zero impact on existing dispatch chain behavior. Immediately useful for trading systems and 24/7 deployments.

---

## Core Feature 2: Fractal Compression

### GPU-Native Compression as a Builtin

**What it is:** Fractal compression and decompression built into OctoFlow as a language-level feature. Not a library. Not an external tool. A builtin that makes every `.flow` program compression-aware, with the GPU handling both compression and decompression as dispatch chains.

**Why it matters:** Current compression workflows are CPU-bound bottlenecks in GPU pipelines. Data arrives compressed → CPU decompresses (slow) → GPU processes (fast) → CPU compresses output (slow) → write to disk. The CPU is the hourglass neck at both ends.

With fractal compression as a builtin, the entire pipeline stays on GPU: load compressed → GPU decompresses via dispatch chain → GPU processes → GPU compresses via dispatch chain → write. No CPU bottleneck. No data transfer between CPU and GPU memory for compression tasks.

### How Fractal Compression Works

Fractal compression exploits self-similarity: patterns that repeat at different scales within the data. Instead of encoding the data directly, it encodes the *transforms* that reproduce the data when applied iteratively.

**Compression (encoding):**
1. Partition data into range blocks (small)
2. Search for matching domain blocks (larger) that, when transformed (scaled, rotated, contrast-adjusted), approximate each range block
3. Encode the transform parameters (which domain block, what affine transform)
4. The encoded transforms are much smaller than the original data

**Decompression (decoding):**
1. Start with any seed data (zeros, random, anything)
2. Apply all transforms iteratively
3. The output converges to the original data — guaranteed by the contractive mapping theorem
4. Typically converges in 8-15 iterations

### Why This Maps Perfectly to OctoFlow

Both stages are massively parallel and map directly to dispatch chains:

**Compression dispatch chain:**
```
partition_kernel        → divide data into range blocks (parallel across all blocks)
  ↓ barrier
search_kernel           → find best domain match per range block (parallel search)
  ↓ barrier  
score_kernel            → rank matches by transform quality (reduce + sort)
  ↓ barrier
encode_kernel           → output transform parameters (parallel encode)
```

**Decompression dispatch chain:**
```
// N iterations, single submit, zero CPU wakeups
apply_transforms_kernel → apply all transforms to current state (parallel)
  ↓ barrier
apply_transforms_kernel → converge further
  ↓ barrier
... repeat N times ...
// GPU runs entire convergence autonomously
```

Decompression is literally what OctoFlow was built for: an autonomous dispatch chain that iterates until convergence. Single submit, GPU runs 8-15 iterations of the transform kernel with pipeline barriers, CPU reads the final result.

**Existing primitives used:**
- Prefix scan — block boundary detection, partition assignment
- Reduce — similarity scoring across candidate blocks
- Sort — rank transform matches by quality
- Push constants — iteration count, convergence threshold, scale level
- Indirect dispatch — variable block sizes at different fractal scales

### The Builtin Integration

```
// Transparent compression — developer doesn't manage it
let data = rt_load_compressed("dataset.fcz")    // GPU decompresses automatically
// ... process data with dispatch chains as normal ...
rt_save_compressed(result, "output.fcz")         // GPU compresses automatically

// Explicit control when needed
let compressed = fractal_compress(buffer, quality: 0.95)
let decompressed = fractal_decompress(compressed, iterations: 12)

// Streaming — decompress as part of a larger dispatch chain
rt_chain_begin()
  rt_dispatch(decompress_pipeline, compressed_input, N_BLOCKS)
  rt_barrier()
  rt_dispatch(process_pipeline, decompressed_buffer, N_ELEMENTS)
  rt_barrier()
  rt_dispatch(compress_pipeline, result_buffer, N_BLOCKS)
rt_chain_end()
// Entire load → decompress → process → compress → save as one chain
```

### The Fractal Foundation

This isn't just a compression feature. It establishes the fractal primitives that flow into everything else:

- **Self-similarity detection** — reusable for pattern recognition, anomaly detection
- **Multi-scale transform encoding** — the same principle drives F-DHGNN's fractal dimension
- **Iterative convergence on GPU** — the dispatch chain pattern for any fixed-point computation
- **Scale-invariant processing** — push constants control scale, same kernel at every level

Fractal compression is the first concrete use case of the fractal foundation. The foundation itself is reusable infrastructure.

### Implementation Scope

- New kernels: partition, search/match, encode, decode/apply-transform (~4-6 kernels)
- Composes heavily from existing keystones (scan, sort, reduce)
- File format: custom `.fcz` with header + transform blocks (simple binary format, similar to existing GIF encoder pattern)
- Estimated size: ~800-1200 lines of `.flow` + kernels
- No runtime changes needed — uses existing dispatch chain infrastructure

### Why It's Second

Medium build scope. Showcases the fractal foundation that feeds into F-DHGNN. Produces a tangible, measurable result (compression ratio + decompression speed). Creates a builtin that makes every OctoFlow program more capable. Builds on the LZ4 decompression kernel (v1.01) that already proved GPU compression is viable.

---

## Core Feature 3: GUI

### GPU-Native Interface Rendering

**What it is:** A GPU-rendered graphical user interface framework written entirely in `.flow`. Not a binding to an existing toolkit. Not CPU-rendered with GPU acceleration bolted on. A GUI where the render pipeline IS a dispatch chain — layout, compositing, text, and presentation all computed on GPU.

**Why it matters:** This transforms OctoFlow from a compute language into a platform. With GUI, OctoFlow programs are no longer terminal-only. They can present data, accept input, display visualizations — all rendered by the same GPU dispatch chains that power the compute.

The long-term trajectory: OctoFlow GUI as foundation for an Electron alternative (desktop apps without shipping a browser) and ultimately a browser alternative (GPU-native content rendering engine).

### Architecture

Four layers, each building on the previous:

**Layer 0: GPU Rendering Primitives**

The lowest level — compute kernels that write pixels to a framebuffer:

- Rect fill (solid, gradient, rounded corners)
- Text rasterization (glyph atlas + parallel glyph placement)
- Line drawing, circles, arcs
- Image blitting and scaling
- Alpha compositing and blending
- Shadow and blur kernels

Each primitive is a compute kernel. The framebuffer is a GPU buffer. Presentation goes through Vulkan's swapchain. OctoFlow already has ray tracing rendering to terminal — Layer 0 upgrades this to pixel-perfect rendering on a proper display surface.

**Layer 1: Widget Toolkit**

The developer-facing API — components that compose into interfaces:

```
// Core widgets
Box         — container with padding, margin, border, background
Text        — styled text with font, size, color, wrapping
Button      — interactive, with hover/press/disabled states
Input       — text input with cursor, selection, clipboard
List        — scrollable list of items
ScrollView  — scrollable container for any content
Image       — display GPU buffers as images
Canvas      — custom draw area for visualizations
```

Layout engine using flexbox-style constraint solving:

```
// Layout as GPU computation
let root = Box(direction: "column", children: [
    Text("OctoFlow Dashboard", size: 24, bold: true),
    Box(direction: "row", children: [
        Canvas(on_draw: render_chart),
        List(items: dispatch_log, height: "fill"),
    ]),
    Text(status_line, color: "gray"),
])

rt_gui_render(root)  // Layout solve + render = one dispatch chain per frame
```

Event handling: input events → parallel hit testing across all widgets → handler dispatch.

**Layer 2: Application Framework**

Window management and system integration:

- Window creation/management via OS FFI (Win32, X11/Wayland, macOS — same pattern as Vulkan surface creation)
- Multi-window support
- File dialogs, clipboard, drag-and-drop
- Component lifecycle (mount, update, unmount)
- State management (reactive updates, minimal re-render)
- Theming and styling system

This is the Electron alternative level — desktop applications built entirely in `.flow`, no browser engine, no JavaScript, no Chromium. The application renders through GPU dispatch chains on a Vulkan surface.

**Layer 3: Content Rendering Engine (Future)**

The long-horizon goal:

- HTML/CSS parser and layout engine (GPU-accelerated)
- Scripting integration
- Network stack
- Standards compliance

This is the browser alternative level. Multi-year, multi-team scope. Documented here for directional clarity, not as a release target.

### Why GPU-Native GUI Is Different

Existing GUI toolkits:

```
CPU computes layout → CPU builds draw commands → GPU executes draw commands
Frame rate limited by CPU layout + command building
```

OctoFlow GUI:

```
GPU computes layout → GPU composites layers → GPU rasterizes text → GPU presents
Frame rate limited only by GPU throughput
```

The layout solver is a compute kernel. Constraint propagation across the widget tree is a dispatch chain with barriers between parent and child resolution. Text shaping is parallel glyph lookup and positioning. Compositing is parallel layer blending. Every stage is a dispatch.

For data-heavy applications (dashboards, monitoring tools, trading interfaces), this means the same GPU that computes the data also renders the visualization of that data — no round-trip through CPU, no serialization between compute and display.

### The Self-Demonstrating Release

The GUI framework's first application: **an OctoFlow dashboard that displays its own performance.** Dispatch chain execution times, GPU temperature (from Homeostasis), kernel utilization, buffer memory usage — all computed on GPU, all rendered on GPU, all in one `.flow` program.

The demo IS the proof. The tool monitors itself using itself.

### Implementation Scope

**Layer 0 (release target):**
- Rendering kernels: rect, text, gradient, composite, blur (~6-8 new kernels)
- Glyph atlas generation from font files
- Vulkan swapchain integration (FFI — surface creation, present)
- Estimated: ~1500-2000 lines of `.flow` + kernels

**Layer 1 (release target):**
- Widget implementations: Box, Text, Button, Input, List, ScrollView
- Layout engine (flexbox subset — direction, alignment, padding, margin, flex-grow)
- Event system (mouse, keyboard → hit test → handler)
- Estimated: ~2000-3000 lines of `.flow`

**Layer 2 (post-release):**
- Window management FFI per platform
- State management, component lifecycle
- Theming
- Estimated: ~2000-4000 lines of `.flow` + platform FFI

**Layer 3 (long-term vision):**
- Scope equivalent to a small browser engine
- Multi-year effort
- Documented for direction, not scheduled

### Why It's Third

Largest build of the three core features. Most visible impact — transforms OctoFlow from compute engine to platform. Requires Layer 0 rendering primitives which benefit from the Mandelbrot and ray tracing work already done. Benefits from Homeostasis (GPU self-regulation during continuous rendering) and Fractal Compression (compressed asset loading).

The build order matters: Homeostasis provides sustainable GPU operation for continuous GUI rendering. Fractal compression provides efficient asset storage and loading. GUI builds on both.

---

## Release Strategy

### Build Order

```
Phase 1: Homeostasis
  - Smallest scope (~500 lines)
  - Immediately useful for existing workloads
  - Novel — nobody has this
  - Can release as standalone update

Phase 2: Fractal Compression
  - Medium scope (~800-1200 lines)
  - Establishes fractal foundation
  - Measurable benchmarks (ratio, speed)
  - Can release as second update

Phase 3: GUI (Layer 0 + Layer 1)
  - Largest scope (~3500-5000 lines)
  - Transforms OctoFlow into a platform
  - Self-demonstrating (dashboard demo)
  - Major release — the "OctoFlow 1.0" moment
```

### Release Options

**Option A: Series of Releases**
Ship each feature as it completes. Community gets incremental updates. Maintains momentum. Risk: each release is smaller and may not individually generate sufficient attention.

**Option B: Single Major Release**
Build all three, ship together as "OctoFlow 1.0." Maximum impact. One launch, one narrative. Risk: longer time to public release, no community feedback during development.

**Option C: Hybrid**
Ship Homeostasis + Fractal Compression as "OctoFlow Preview" to early adopters. Build GUI based on feedback. Ship full platform release as 1.0. Balances momentum with impact.

### What Ships, What Stays Internal

| Component | Ships | Stays Internal |
|---|---|---|
| Self-hosted compiler (<500 lines Rust) | ✅ | |
| 41 GPU kernels | ✅ | |
| 32 verified stdlib functions | ✅ | |
| 5 BIT EXACT keystone primitives | ✅ | |
| 9 domain demos | ✅ | |
| Autonomous agent demos | ✅ | |
| Homeostasis | ✅ | |
| Fractal compression | ✅ | |
| GUI (Layer 0 + 1) | ✅ | |
| 70 experimental stdlib functions | | ✅ until verified |
| THE_CONVERGENCE.md | | ✅ north star |
| F-DHGNN / OctoBrain plans | | ✅ until built |
| GUI Layer 2-3 | | ✅ post-release |

---

## After Core: Return to Convergence

With Homeostasis, Fractal Compression, and GUI complete, OctoFlow is a self-sufficient GPU-native platform:

- A language that compiles itself
- A runtime that regulates itself
- Compression that's native to the pipeline
- A GUI that renders through dispatch chains
- All on vendor-independent consumer hardware

Then the convergence work begins:

```
GEMM kernel              → matrix-matrix for attention
Softmax                  → compose from existing reductions
Embedding lookup         → gather kernel
Runtime compilation API  → .flow compiler as callable library
F-DHGNN Shell            → perception layer on GPU
F-DHGNN Engine           → hypergraph message passing as dispatch chains
F-DHGNN Plasticity       → topology mutation at GPU speed
OctoBrain on OctoFlow    → the brain runs on its own runtime
```

The three core features aren't detours from convergence. They're the final infrastructure convergence requires:

- **Homeostasis** keeps the GPU healthy during OctoBrain's continuous inference-time adaptation
- **Fractal compression** provides the multi-scale transform primitives that F-DHGNN's fractal dimension uses
- **GUI** provides the interface for visualizing and interacting with OctoBrain's dynamic topology

Everything connects. Nothing is wasted.

---

*From Octopus Ontology to breaking barriers.*
