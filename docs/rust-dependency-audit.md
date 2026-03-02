# OctoFlow — Rust Dependency Audit & Refactoring Plan

**Date:** February 17, 2026
**Status:** Post-Phase 42 (801 tests)
**Purpose:** Audit all Rust dependencies, identify what can move to .flow, plan refactoring

---

## Current Rust Dependencies

### flowgpu-cli (The Main Compiler)

```toml
[dependencies]
flowgpu-parser.workspace = true      # Internal (parser crate)
flowgpu-spirv.workspace = true       # Internal (SPIR-V emission)
flowgpu-vulkan.workspace = true      # Internal (GPU runtime)
image = "0.25"                       # External: PNG/JPEG I/O
serde_json = "1"                     # External: JSON parsing
ureq = "2"                           # External: HTTP client
base64 = "0.22"                      # External: Base64 encoding
time = "0.3"                         # External: Date/time parsing
```

**Total external dependencies: 5** (image, serde_json, ureq, base64, time)

### flowgpu-vulkan (GPU Runtime)

```toml
[dependencies]
ash = "0.38"                         # External: Vulkan bindings
flowgpu-spirv.workspace = true       # Internal
```

**Total external dependencies: 1** (ash)

### flowgpu-spirv, flowgpu-parser (Zero Dependencies ✅)

```toml
[dependencies]
# None — pure Rust
```

**This is good.** Parser and SPIR-V emitter have no external dependencies.

### Summary

**Total external Rust crates: 6**
- ash (Vulkan bindings) — 1 crate
- image (PNG/JPEG) — 1 crate
- serde_json (JSON) — 1 crate
- ureq (HTTP) — 1 crate
- base64 (encoding) — 1 crate
- time (date/time) — 1 crate

**Total lines of external Rust code (estimated):**
- ash: ~5,000 lines (thin Vulkan wrapper)
- image: ~15,000 lines (PNG/JPEG codecs)
- serde_json: ~8,000 lines
- ureq: ~6,000 lines
- base64: ~1,000 lines
- time: ~1,500 lines

**Total: ~36,500 lines of external Rust dependencies**

---

## Dependency Analysis: Can It Move to .flow?

| Dependency | Lines | Purpose | Can Move to .flow? | When? | Strategy |
|------------|-------|---------|-------------------|-------|----------|
| **ash** | ~5K | Vulkan C API bindings | **YES** ✅ | Phase 50-51 | **extern FFI declarations in .flow** (like Python/Go/Julia do) |
| **image** | ~15K | PNG/JPEG decode/encode | **YES** | Phase 54+ | Implement PNG/JPEG codecs in .flow OR use FFI wrapper |
| **serde_json** | ~8K | JSON parse/stringify | **YES** | Phase 45-46 | JSON parser in .flow (recursive descent on JSON grammar) |
| **ureq** | ~6K | HTTP client | **YES** | Phase 45 | HTTP = TCP sockets + protocol parsing (both in .flow) |
| **base64** | ~1K | Base64 encode/decode | **YES** | Phase 43-44 | Algorithm is simple (lookup table), ~100 lines in .flow |
| **time** | ~1.5K | ISO8601 parsing | **YES** | Phase 50 | Use regex + date algorithms (already designed for this) |

**VERDICT: ALL 6 dependencies can move to .flow. ZERO Rust dependencies after Phase 52.**

**Critical insight:** Vulkan is just a C shared library (vulkan-1.dll / libvulkan.so). Python, Go, Haskell, Common Lisp, Julia all call it via FFI. **OctoFlow can too** — via `extern` declarations in .flow (like LuaJIT's ffi.cdef or Julia's @ccall).

**The only Rust that must exist: ~2,000 line bootstrap interpreter to run the first .flow compiler.**

---

## Refactoring Priority

### Phase 43-44: Base64 in .flow (IMMEDIATE)

**Current:** Uses `base64` crate (1,000 lines of Rust)

**Refactor to .flow:**
```flow
// stdlib/encoding.flow

let base64_chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"

fn base64_encode(input: string) -> string {
    let bytes = string_to_bytes(input)
    let mut result = ""

    let mut i = 0
    while i < len(bytes)
        let b1 = bytes[i]
        let b2 = if i+1 < len(bytes) then bytes[i+1] else 0
        let b3 = if i+2 < len(bytes) then bytes[i+2] else 0

        let c1 = (b1 >> 2) & 0x3F
        let c2 = ((b1 & 0x03) << 4) | ((b2 >> 4) & 0x0F)
        let c3 = ((b2 & 0x0F) << 2) | ((b3 >> 6) & 0x03)
        let c4 = b3 & 0x3F

        result = result + char_at(base64_chars, c1)
        result = result + char_at(base64_chars, c2)
        result = result + (if i+1 < len(bytes) then char_at(base64_chars, c3) else "=")
        result = result + (if i+2 < len(bytes) then char_at(base64_chars, c4) else "=")

        i = i + 3
    end

    return result
}
```

**Requirements:** Byte manipulation (Phase 49 adds byte type), bitwise ops (need >> << & | operators)

**When:** Phase 49 (after byte type exists)

### Phase 45-46: JSON Parser in .flow

**Current:** Uses `serde_json` crate (8,000 lines of Rust)

**Refactor to .flow:**
```flow
// stdlib/json_parser.flow

fn json_parse(input: string) -> Map {
    let tokens = tokenize_json(input)  // Use regex from Phase 43
    return parse_json_object(tokens, 0).value
}

fn parse_json_object(tokens: [Token], pos: u32) -> {value: Map, end_pos: u32} {
    // Recursive descent parser for JSON
    // Similar to .flow parser but simpler (JSON grammar is simpler)
}
```

**Requirements:** Regex (Phase 43 ✅), recursive functions (already have ✅), hashmaps (already have ✅)

**When:** Phase 45-46 (after regex + pattern matching mature)

### Phase 45-46: HTTP Client in .flow

**Current:** Uses `ureq` crate (6,000 lines of Rust)

**Refactor to .flow:**
```flow
// stdlib/http.flow

fn http_get(url: string) -> {status: float, body: string, ok: float, error: string} {
    // Parse URL
    let parts = parse_url(url)  // Use regex

    // Open TCP socket
    let socket = tcp_connect(parts.host, parts.port)

    // Send HTTP request
    let request = "GET {parts.path} HTTP/1.1\r\nHost: {parts.host}\r\n\r\n"
    tcp_send(socket, request)

    // Read response
    let response = tcp_recv(socket)
    tcp_close(socket)

    // Parse HTTP response
    return parse_http_response(response)
}
```

**Requirements:** TCP sockets (Phase 45), regex (Phase 43 ✅), string ops (already have ✅)

**When:** Phase 45 (when TCP sockets exist)

### Phase 50+: ISO8601 Parser in .flow

**Current:** Uses `time` crate (1,500 lines) for ISO8601 parsing

**Refactor to .flow:**
```flow
// stdlib/time.flow

fn timestamp(iso_string: string) -> float {
    // Parse "2024-01-15T13:30:00Z"
    let pattern = "(\\d{4})-(\\d{2})-(\\d{2})T(\\d{2}):(\\d{2}):(\\d{2})Z"
    let parts = regex_match(iso_string, pattern)

    let year = int(parts[0])
    let month = int(parts[1])
    let day = int(parts[2])
    let hour = int(parts[3])
    let minute = int(parts[4])
    let second = int(parts[5])

    return date_to_unix(year, month, day, hour, minute, second)
}

fn date_to_unix(year: float, month: float, day: float, hour: float, minute: float, second: float) -> float {
    // Algorithm to convert calendar date to unix timestamp
    // Days since epoch + time of day
    // This is ~50 lines of date arithmetic
}
```

**Requirements:** Regex (Phase 43 ✅), int() conversion (already have ✅)

**When:** Phase 50 (when parser is .flow, can self-host this too)

### Phase 54+: PNG/JPEG in .flow (MAYBE)

**Current:** Uses `image` crate (15,000 lines)

**Refactor options:**
1. **Implement PNG/JPEG codecs in .flow** (hard, ~5,000 lines each)
2. **FFI wrapper** — Rust FFI for decode/encode, .flow for everything else
3. **Keep as Rust** — codecs are complex, not core to self-hosting

**Recommendation:** Option 2 or 3. PNG/JPEG codecs are not critical to self-hosting. Focus on compiler first.

---

## Refactoring Timeline

### Phase 43-44: Planning

- Document which functions will move to .flow
- Design stdlib/compiler/ directory structure
- Prepare for preflight/lint as data

### Phase 45-46: First Migrations

- **JSON parser → .flow** (`stdlib/json_parser.flow`)
- **HTTP client → .flow** (`stdlib/http.flow`)
- Remove serde_json and ureq dependencies

**Impact:** -2 dependencies, -14,000 lines of external Rust

### Phase 49: Base64 in .flow

- **Base64/hex → .flow** (`stdlib/encoding.flow`)
- Remove base64 dependency

**Impact:** -1 dependency, -1,000 lines of external Rust

### Phase 50: ISO8601 Parser in .flow

- **Date parsing → .flow** (`stdlib/time.flow`)
- Keep time crate for complex formats OR remove entirely

**Impact:** -1 dependency (maybe), -1,500 lines of external Rust

### Phase 52 (Public Release)

**Remaining external dependencies: 2**
- `ash` (Vulkan — stays forever)
- `image` (PNG/JPEG — decision deferred)

**External Rust: ~20,000 lines** (down from ~36,500)

**Compiler/stdlib in .flow: ~25,000 lines**

**Ratio: 1:1.25** (more .flow than external Rust)

---

## Current vs Target Architecture

### Current (Phase 42)

```
OctoFlow Binary:
├── Rust crates (~50,000 lines)
│   ├── flowgpu-cli (~15,000 lines)
│   ├── flowgpu-parser (~2,000 lines)
│   ├── flowgpu-spirv (~2,000 lines)
│   ├── flowgpu-vulkan (~3,000 lines)
│   └── External deps (~36,500 lines)
│       ├── ash (Vulkan)
│       ├── image (PNG/JPEG)
│       ├── serde_json
│       ├── ureq
│       ├── base64
│       └── time
└── stdlib/ (~500 lines .flow)
    ├── math.flow
    ├── array_utils.flow
    └── sort.flow

Ratio: 100:1 Rust-heavy
```

### Target (Phase 52 Public Release)

```
OctoFlow Binary:
├── Rust runtime kernel (~10,000 lines)
│   ├── bootstrap.rs (minimal .flow interpreter)
│   ├── spirv_ffi.rs (SPIR-V emission FFI)
│   ├── vulkan.rs (GPU runtime)
│   ├── system.rs (file/network/exec FFI)
│   └── External deps (~20,000 lines)
│       ├── ash (Vulkan — MUST stay)
│       └── image (PNG/JPEG — MAYBE stay)
│
└── stdlib/ (~25,000 lines .flow)
    ├── compiler/
    │   ├── parser.flow (~2,000 lines)
    │   ├── preflight.flow (~1,000 lines)
    │   ├── lint.flow (~800 lines)
    │   ├── typechecker.flow (~2,000 lines)
    │   ├── optimizer.flow (~1,500 lines)
    │   └── codegen.flow (~3,000 lines)
    ├── json_parser.flow (~800 lines)
    ├── http.flow (~500 lines)
    ├── encoding.flow (~200 lines)
    ├── time.flow (~500 lines)
    ├── ml/
    │   ├── hypergraph.flow (~2,000 lines)
    │   ├── gnn.flow (~1,500 lines)
    │   └── autograd.flow (~1,000 lines)
    └── (existing stdlib)

Ratio: 1:2.5 (.flow-heavy — 2.5x more .flow than Rust)
```

---

## Immediate Action Items (Phase 43)

Before implementing Phase 43 features, **design for self-hosting:**

1. **Bitwise operators** (>>, <<, &, |, ^)
   - Needed for: base64 encoding, byte manipulation
   - Implementation: eval_scalar BinOp extension (~50 lines Rust)
   - Later: moves to .flow when compiler is .flow

2. **Byte type** (Phase 49 accelerated to Phase 43?)
   - Needed for: binary data, encoding, network protocols
   - Representation: Value::Byte(u8) OR array of u8
   - Later: critical for self-hosting (SPIR-V is bytes)

3. **Pattern for .flow stdlib calling Rust FFI**
   - Example: `stdlib/encoding.flow` calls `base64_encode_ffi()` Rust function
   - This allows gradual migration (high-level logic in .flow, low-level in Rust FFI)
   - Clean separation: stdlib/ is .flow, runtime/ is Rust

---

## Refactoring Principles

### 1. Pure Algorithms → .flow First

Functions that are pure math/logic with no external dependencies:
- Date arithmetic (add_days, diff_seconds) ✅ Already pure, can move to .flow immediately
- Statistics (mean, median, stddev) ✅ Already pure
- Path operations (join_path, dirname) ✅ String operations, pure

**These can be in stdlib/ right now** if we want. They're calling Rust only because stdlib doesn't have a way to be evaluated yet.

### 2. External Dependencies → Evaluate

For each external crate, ask:
- **Can the algorithm be implemented in .flow?** (serde_json → yes, recursive descent parser)
- **Is performance critical?** (image codecs → yes, but can use FFI)
- **Is it low-level system interface?** (ash → yes, must stay Rust)

### 3. FFI Boundary Pattern

When we can't eliminate Rust entirely:
```
.flow high-level API → calls → Rust FFI minimal wrapper → calls → external crate

Example:
  stdlib/image.flow:
    fn load_png(path: string) -> Tensor {
        return png_decode_ffi(path)  // Thin FFI to Rust image crate
    }

  runtime/image_ffi.rs:
    fn png_decode_ffi(path: &str) -> Vec<f32> {
        image::open(path)...  // Delegate to image crate
    }
```

This isolates dependencies in runtime/, keeps stdlib/ pure .flow.

---

## Recommended Immediate Changes (Phase 43)

### 1. Create `stdlib/encoding.flow` (Move base64 Logic)

Even though current implementation calls Rust `base64` crate, **wrap it in a .flow stdlib module:**

```flow
// stdlib/encoding.flow (Phase 43)

// These call Rust FFI for now, but API is .flow
fn base64_encode(input: string) -> string {
    return base64_encode_ffi(input)  // FFI to runtime/encoding.rs
}

fn base64_decode(input: string) -> string {
    return base64_decode_ffi(input)  // FFI to runtime/encoding.rs
}

// Later (Phase 49+): replace FFI with pure .flow implementation
```

**Benefit:** Public API is .flow. Users import `stdlib/encoding`, not Rust functions. When we implement base64 in pure .flow, we just replace the FFI calls — API stays same.

### 2. Create `stdlib/time.flow` (Move Date Functions)

```flow
// stdlib/time.flow (Phase 43)

// Pure math functions (NO Rust dependency, can move today)
fn add_seconds(ts: float, seconds: float) -> float {
    return ts + seconds
}

fn add_days(ts: float, days: float) -> float {
    return ts + (days * 86400.0)
}

fn diff_seconds(ts1: float, ts2: float) -> float {
    return ts1 - ts2
}

// Parsing (uses Rust FFI for now)
fn timestamp(iso_string: string) -> float {
    return timestamp_parse_ffi(iso_string)  // FFI to runtime/time.rs
}

fn now() -> float {
    return now_ffi()  // FFI to runtime/time.rs
}

// Later (Phase 50): replace parse FFI with .flow regex implementation
```

**Benefit:** Date arithmetic is already pure .flow. Only parsing uses FFI. Clean separation.

### 3. Move Existing Rust Implementations to be FFI-Wrapped

**Pattern:**
1. Move function implementation from `compiler.rs` to `runtime/time_ffi.rs`
2. Expose as FFI function
3. Call from `stdlib/time.flow`
4. Compiler loads stdlib and inlines the .flow functions

**This can happen incrementally.** Start with new features in stdlib/, gradually migrate existing ones.

---

## Dependency Reduction Roadmap

| Phase | Action | Dependencies Removed | Lines Saved |
|-------|--------|---------------------|-------------|
| 43-44 | Base64 → .flow (pure algorithm) | base64 crate | -1,000 |
| 45-46 | JSON parser → .flow | serde_json | -8,000 |
| 45-46 | HTTP client → .flow | ureq | -6,000 |
| 50 | ISO8601 parser → .flow (regex-based) | time crate | -1,500 |
| 54+ | PNG/JPEG → .flow OR keep FFI | (TBD) | 0 or -15,000 |

**Total potential savings: ~31,500 lines of external Rust**

**After Phase 50: Only ash (Vulkan) + maybe image remain**

---

## Self-Hosting Integration

When the compiler moves to .flow (Phases 48-52), stdlib functions automatically become part of the self-hosting story:

```
.flow compiler uses:
├── stdlib/time.flow (for timestamp operations in compiler)
├── stdlib/encoding.flow (for module hashing, signatures)
├── stdlib/json_parser.flow (for flow.project parsing)
└── stdlib/compiler/parser.flow (to parse itself!)
```

**The compiler dogfoods the stdlib.** Any stdlib limitation affects the compiler → gets fixed immediately.

---

## Summary & Next Steps

**Current state:**
- 6 external Rust dependencies (~36,500 lines)
- Ratio: 100:1 Rust-heavy

**Target (Phase 52):**
- 1-2 external Rust dependencies (~5,000-20,000 lines)
- Ratio: 1:2.5 (.flow-heavy)

**Next steps:**
1. **Phase 43:** Add bitwise operators (>>, <<, &, |, ^) for encoding
2. **Phase 43:** Create `stdlib/encoding.flow` with FFI wrapper pattern
3. **Phase 43:** Create `stdlib/time.flow` with pure .flow arithmetic
4. **Phase 45-46:** Implement JSON parser in .flow, remove serde_json
5. **Phase 45-46:** Implement HTTP client in .flow, remove ureq
6. **Phase 50:** Implement ISO8601 parser in .flow, remove time crate

**Principle:** Every new feature should ask: "Can this be .flow?" If yes, make it .flow (even if it calls FFI initially). If no, isolate it in runtime/ FFI.

---

*"Rust is the scaffold. .flow is the building. By Phase 52, you see mostly .flow."*
