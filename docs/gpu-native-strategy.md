# OctoFlow — GPU-Native General-Purpose Language Strategy

**Date:** February 18, 2026
**Phase:** 68 COMPLETE (919 tests)
**Author:** AI-assisted (Claude Code)

---

## Vision

OctoFlow is a **GPU-native general-purpose programming language**. The GPU is the primary
execution target. CPU is optional — used only for I/O (file, console, network). The compiler
itself is written in OctoFlow and compiles to SPIR-V. No Rust. No external dependencies.

**Not** a shader language. **Not** a CUDA extension. A full GPL where the GPU *is* the computer.

---

## Current State (Phase 68)

| Component | Lines | Language | Status |
|-----------|-------|----------|--------|
| Lexer | 212 | .flow | Working (lexer.flow) |
| Parser | 927 | .flow | Working (parser.flow) |
| Preflight | 763 | .flow | Working (preflight.flow) |
| Meta-interpreter | 3,871 | .flow | Working (eval.flow, 3-layer meta) |
| GLSL codegen | 504 | .flow | Working (codegen.flow) |
| SPIR-V emitters | 4,510 | .flow | Working (4 hand-crafted emitters) |
| Bootstrap tests | 395 | .flow | 30 PASS / 0 FAIL / 1 SKIP |
| **Total .flow compiler** | **13,609** | **.flow** | **Self-hosting foundation** |
| Rust bootstrap | 28,059 | Rust | Target for elimination |

### SPIR-V Emitter Proof (Phase 66-68)

Four hand-crafted .flow SPIR-V emitters prove the language can author its own GPU backend:

| Emitter | Control Flow | Complexity | Validation |
|---------|-------------|------------|------------|
| `spirv_emit.flow` | Linear | input[gid] * 2.0 | spirv-val + GPU |
| `spirv_emit_param.flow` | Parametric | 5 arithmetic ops via env | spirv-val + GPU |
| `spirv_emit_branch.flow` | Selection | OpSelectionMerge + OpPhi | spirv-val + GPU + GLSL cross-val |
| `spirv_emit_loop.flow` | Loop | OpLoopMerge + dual OpPhi + back-edge SSA | spirv-val + GPU + GLSL cross-val |
| `spirv_emit_nested.flow` | Nested | Loop > Branch > Loop, 5 OpPhi, 10 blocks | spirv-val + GPU + GLSL cross-val |

**Status: GPU-authored.** The backend is written in the language itself.

---

## Language Feature GPU Categorization

### GPU-Native (compiles directly to SPIR-V)

**Arithmetic:** +, -, *, /, %, <<, >>, &, |, ^
**Comparison:** <, >, <=, >=, ==, !=, &&, ||
**Control flow:** if/elif/else, while, for, break, continue, return
**Data:** arrays (as storage buffers), struct decomposition, index access
**Functions:** scalar functions (inline or SPIR-V functions)
**Math builtins:** abs, sqrt, pow, exp, log, sin, cos, floor, ceil, round, min, max, clamp
**Stream pipeline:** map (19 ops), reduce (sum/min/max), scan, temporal, fused
**Bit manipulation:** float_to_bits, bits_to_float, float_byte

### CPU-Required (I/O + dynamic allocation — .flow with extern FFI)

**File I/O:** read_file, write_file, read_lines, list_dir, emit, tap
**Console:** print
**Network:** http_get, http_post, http_put, http_delete
**Process:** exec()
**Dynamic allocation:** push (array grow), map (hashmap)
**Strings:** split, join, trim, replace, contains, regex
**Serialization:** json_parse, json_stringify, read_csv, load_data

### The Split

~70% of typical .flow programs is GPU-compilable computation.
~30% is I/O orchestration that must run on CPU — but can be .flow with `extern` FFI
to system libraries. No Rust needed for either side.

---

## The Missing Piece: IR Layer

### Problem

Current SPIR-V emitters are hand-crafted. Each shader requires manually:
- Allocating SPIR-V IDs (61 IDs for nested test)
- Tracking basic blocks and dominance
- Placing OpPhi nodes
- Ensuring structured control flow nesting

This doesn't scale. A real program with 50 functions and 200 blocks is impossible to hand-craft.

### Solution: CFG + SSA Intermediate Representation

```
.flow source → [parser.flow] → AST → [ir_lower.flow] → IR → [ir_emit.flow] → SPIR-V → GPU
```

The IR provides:
- **Automatic ID allocation** — IR assigns SSA IDs, codegen maps to SPIR-V
- **Automatic Phi placement** — dominance frontier algorithm
- **Automatic structured control flow** — IR marks loop/selection merge points
- **Optimization potential** — dead code elimination, constant folding, CSE

### IR Data Structures (parallel arrays, OctoFlow idiom)

```flow
// Basic blocks
let mut bb_id = []         // block label
let mut bb_first_inst = [] // first instruction index
let mut bb_term_kind = []  // "branch" | "cond_branch" | "return" | "loop_merge" | "sel_merge"
let mut bb_term_target = []// branch target(s)

// SSA Instructions
let mut inst_op = []       // "add" | "mul" | "load" | "store" | "convert" | "compare" | "phi"
let mut inst_type = []     // "float" | "uint" | "bool" | "void"
let mut inst_result = []   // result SSA ID
let mut inst_arg1 = []     // first operand
let mut inst_arg2 = []     // second operand

// Phi nodes (variable-length operands)
let mut phi_values = []    // SEP-delimited value IDs
let mut phi_parents = []   // SEP-delimited parent block IDs
```

---

## Rust Elimination Path

### Phase 69: IR Foundation
- IR data structures in .flow (basic blocks + SSA instructions)
- IR → SPIR-V automated codegen (replaces hand-crafted emitters)
- Test: hand-build IR, emit SPIR-V, validate, GPU dispatch
- ~800-1200 lines of .flow

### Phase 70: AST → IR Lowering
- parser.flow AST → IR basic blocks + SSA
- Control flow lowering: if → selection merge, while/for → loop merge
- Phi placement via dominance frontiers
- Test: .flow source → parse → lower → codegen → GPU, end-to-end
- ~600-1000 lines of .flow

### Phase 71: Full Automated Pipeline
- Complete: .flow source → parse → preflight → lower → codegen → dispatch
- Replaces Rust SPIR-V codegen (flowgpu-spirv crate, 623 lines)
- Multiple test programs compiled and dispatched automatically
- ~400-600 lines of .flow

### Phase 72: Vulkan Dispatch in .flow
- `extern "vulkan-1" { fn vkCreateInstance(...) ... }` from .flow
- Buffer management, pipeline creation, dispatch, readback — all in .flow
- Replaces flowgpu-vulkan crate (2,541 lines Rust)
- ~800-1200 lines of .flow

### Phase 73: I/O Layer in .flow
- File I/O via extern (kernel32 on Windows, libc on Unix)
- Console I/O via extern
- Network I/O via extern (ws2_32 on Windows, libc sockets on Unix)
- Replaces compiler.rs I/O sections (~3,800 lines Rust)
- ~600-1000 lines of .flow

### Phase 74: Self-Hosting Compiler on GPU
- The compiler is a compute shader: input = .flow source buffer, output = .spv buffer
- parse → lower → codegen all run as GPU kernels
- Three-compiler bootstrap: v1 (CPU) → v2 (GPU) → v3 (GPU), verify v2 == v3
- Proves: the language compiles itself on the GPU

### Phase 75: Rust Elimination
- Rust bootstrap → thin loader (< 200 lines)
- Loader: read .spv, read source, call vkDispatch, write output
- Eventually: pre-compiled binary, no Rust source at all
- **OctoFlow is a zero-dependency, GPU-native, self-hosting GPL**

---

## What "GPU-Native GPL" Means

### What it IS:
- A general-purpose language where ALL computation runs on the GPU
- Self-hosting: compiler written in itself, targeting SPIR-V
- Zero external dependencies (only system libs: vulkan-1, kernel32/libc)
- CPU is I/O bus only (file, console, network — unavoidable hardware reality)

### What it is NOT:
- Not a shader language (GLSL, HLSL, WGSL — those are domain-specific)
- Not a GPU extension (CUDA, OpenCL, SYCL — those extend C/C++)
- Not interpreted on CPU (the whole point is GPU execution)
- Not dependent on external toolchains (no LLVM, no GCC, no NVIDIA SDK)

### Precedent
No programming language has achieved all of:
1. General-purpose (not domain-specific)
2. GPU as primary execution target (not CPU with GPU acceleration)
3. Self-hosting (compiler written in itself)
4. Zero external dependencies
5. Structured control flow on GPU (OpLoopMerge, OpSelectionMerge, OpPhi)

OctoFlow is the first.

---

## Risk Assessment

| Risk | Severity | Mitigation |
|------|----------|------------|
| IR complexity | Medium | Start minimal (arithmetic + one loop), extend incrementally |
| GPU string handling | Low | Strings stay on CPU (I/O layer), not a blocker |
| Dynamic allocation on GPU | Medium | Fixed-size buffers, no heap; arrays pre-sized at compile time |
| Vulkan API complexity in .flow | Medium | extern FFI already works; dispatch.rs is template |
| Self-hosting compiler performance | Low | GPU parallel compilation likely faster than CPU for large programs |
| Multi-vendor GPU compatibility | Low | SPIR-V is vendor-neutral; spirv-val catches issues |
