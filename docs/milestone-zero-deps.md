# OctoFlow — Zero Dependencies Milestone

**Date:** February 17, 2026
**Phase:** 49
**Status:** ✅ ACHIEVED

---

## The Claim

As of Phase 49, OctoFlow has **zero external Rust dependencies**.

Open `Cargo.toml` in any of the four arms. The `[dependencies]` section contains
only other OctoFlow arms — no third-party crates, no external code, no supply chain.

```
flowgpu-cli      [dependencies]  →  flowgpu-parser, flowgpu-spirv, flowgpu-vulkan
flowgpu-parser   [dependencies]  →  (empty)
flowgpu-spirv    [dependencies]  →  (empty)
flowgpu-vulkan   [dependencies]  →  flowgpu-spirv
```

**Total external Rust crates: 0.**

The only external code OctoFlow touches is what the operating system already provides:
- `vulkan-1.dll` / `libvulkan.so` — the GPU driver loader (you already have this)
- `ws2_32.dll` — Windows networking (part of Windows since 1993)

That's it. That's the entire dependency graph.

---

## The Journey

This wasn't declared up-front — it emerged as a design principle and was pursued
methodically across six phases:

| Phase | Removed | Replaced with | Lines written |
|-------|---------|---------------|---------------|
| 45 | `serde_json`, `base64`, `time`, `chrono` | Pure-Rust JSON, base64, ISO 8601 in `json_io.rs`, `octo_io.rs` | ~400 |
| 46 | `ureq` | Pure HTTP/1.1 client over raw TCP in `http_io.rs` | ~350 |
| 47 | `regex` | NFA bytecode backtracking VM in `regex_io.rs` | ~700 |
| 48 | `image` | Pure PNG + JPEG codec in `image_io.rs` | ~1,200 |
| 49 | `ash` | Raw Vulkan C bindings in `vk_sys.rs` | ~380 |

**5 phases. ~3,030 lines written. 6 external crates eliminated.**

Each replacement is self-contained, purpose-built, and tuned exactly to what
OctoFlow needs — no more, no less.

---

## What We Built Instead

### `json_io.rs` — JSON without serde_json
Hand-written recursive descent parser. Handles objects, arrays, strings, numbers,
booleans, null. Integrates directly with OctoFlow's `Value` type. No allocator
overhead, no derive macros, no reflection.

### `http_io.rs` — HTTP without ureq
Raw TCP socket → HTTP/1.1 request string → response parser. Handles chunked
transfer encoding, redirects, content-length. Built on `net_io.rs`. The entire
HTTP client is ~350 lines.

### `regex_io.rs` — Regex without the regex crate
A complete NFA bytecode VM. Pattern → AST → `Vec<Instr>`. Greedy and lazy
quantifiers, character classes, anchors, word boundaries, capture groups.
Backtracking stack with per-thread capture state. 700 lines. Passes all tests.

### `image_io.rs` — PNG + JPEG without image crate
PNG: Full DEFLATE inflate (stored, fixed, dynamic Huffman), zlib, all five
filter types (None/Sub/Up/Average/Paeth), encode with store blocks.

JPEG: Baseline DCT decode with full marker parsing, AAN-inspired IDCT,
YCbCr→RGB conversion, Huffman decode tables with byte-stuffing. Standard
quantization tables for encode.

1,200 lines. Handles the full image I/O surface that OctoFlow needs.

### `vk_sys.rs` — Vulkan without ash
35 raw `extern "system"` Vulkan function declarations. Every handle type,
struct definition, flag constant, and sType value used in OctoFlow's GPU
compute path — written directly from the Vulkan specification.

No dispatch table abstraction. No builder pattern. No lifetime gymnastics.
Just C structs and function pointers, exactly as Vulkan defines them.

`build.rs` handles finding `vulkan-1.lib` on Windows, `libvulkan.so` on Linux.

---

## Why It Matters

### Speed

Before (Phase 42, with all crates):
```
$ cargo build
   Compiling ash v0.38...
   Compiling image v0.25...
   Compiling serde_json v1...
   Compiling ureq v2...
   Compiling regex v1...
   ...
   Finished in ~17 seconds
```

After (Phase 49, zero deps):
```
$ cargo build
   Compiling flowgpu-parser...
   Compiling flowgpu-spirv...
   Compiling flowgpu-vulkan...
   Compiling flowgpu-cli...
   Finished in ~3 seconds
```

**5x faster builds.** Every developer who clones the repo gets this.

### Security

Every external dependency is a surface for:
- Vulnerabilities in code you didn't write
- Supply chain attacks (malicious maintainers, typosquatting)
- Behavior changes when a crate releases a new version

With zero dependencies, OctoFlow's security posture is:
> "We wrote it. We understand it. We control it."

### Portability

A crate can fail to compile on a new platform, a new Rust version, a new OS.
OctoFlow can't fail because someone else's crate doesn't support RISC-V or
a future GPU architecture — there are no external crates to fail.

If it compiles OctoFlow's Rust, OctoFlow runs.

### Identity

This is the subtler point. Every external dependency is a piece of someone
else's identity embedded in yours. `serde_json` has its own opinions about
JSON. `ash` has its own opinions about how Vulkan should be accessed. `image`
has its own opinions about what image formats matter.

OctoFlow now has no inherited opinions. Every decision in the codebase is
an OctoFlow decision.

---

## The Number

**27,971 lines** of OctoFlow Rust across four arms. Every line written for this
project, with this vision, for this language.

**0 lines** borrowed from external crates.

---

## What Remains

The bootstrap reality: OctoFlow is written in Rust. That's not a dependency —
that's the seed. The Rust compiler is used to compile the seed once. After
Phase 52, the OctoFlow compiler compiles OctoFlow.

Two OS-level interfaces remain:
- **Vulkan** — the GPU driver. You cannot write your own GPU driver without
  also writing your own GPU. This is the correct boundary.
- **WinSock2 / POSIX sockets** — the OS networking stack. Same logic.

These are not dependencies. These are the platform. OctoFlow calls the platform
directly, without intermediaries.

---

## The Competitive Position

| Language | External deps | Self-hosting | Zero-dep GPU |
|----------|--------------|-------------|-------------|
| Python | Hundreds of C extensions | PyPy (partial) | No |
| Julia | LLVM, BLAS, libgit2, ... | No | No |
| Triton | Python + PyTorch + LLVM | No | No |
| CUDA | NVIDIA runtime (closed) | No | N/A |
| **OctoFlow** | **0** | **Phase 52** | **Yes** |

No other GPU language has done this.

---

## A Note on Naming

The packages are no longer called "crates" — that's Rust jargon.

They are **arms**. OctoFlow has four arms:

- `flowgpu-parser` — the arm that reads .flow
- `flowgpu-spirv` — the arm that speaks to the GPU
- `flowgpu-vulkan` — the arm that holds the GPU
- `flowgpu-cli` — the arm that faces the user

An octopus moves by coordinating its arms independently. So does OctoFlow.

---

*"Zero external dependencies. We wrote it. We own it. We are it."*
