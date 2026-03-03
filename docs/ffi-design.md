# OctoFlow FFI Design (Phase 43-44)

**Purpose:** Enable .flow code to call C shared libraries (Vulkan, libc, custom .so/.dll)
**Goal:** Zero external Rust dependencies by Phase 52

---

## Design Principles

1. **Simple syntax** — looks like regular function calls
2. **Type-safe** — compiler knows C signatures
3. **Portable** — works on Windows/Linux/macOS
4. **Minimal overhead** — direct function pointer calls
5. **Self-hosting friendly** — FFI layer itself can be .flow (eventually)

---

## Proposed Syntax

### Declaring External Functions

```flow
// Declare functions from a C shared library
extern "vulkan-1" {
    fn vkCreateInstance(create_info: ptr, allocator: ptr, instance: ptr) -> u32
    fn vkEnumeratePhysicalDevices(instance: u64, count: ptr, devices: ptr) -> u32
}

extern "c" {
    fn strlen(s: ptr) -> u64
    fn malloc(size: u64) -> ptr
    fn free(p: ptr)
}
```

### Calling External Functions

```flow
// Call like normal functions
let instance: u64 = 0
let result = vkCreateInstance(&create_info, null, &instance)

if result == 0  // VK_SUCCESS
    print("Vulkan instance created: {instance}")
end
```

---

## Type Mapping

### C Types → .flow Types

| C Type | .flow Type | Notes |
|--------|------------|-------|
| int, int32_t | float | Store as f32, cast when passing |
| uint32_t, uint64_t | float | Store as f32/f64, cast when passing |
| float, double | float | Direct mapping |
| char*, const char* | string | Null-terminated conversion |
| void* | ptr | Opaque pointer type |
| struct Foo* | ptr | Pointer to struct |
| uint8_t | byte (Phase 44) | Single byte value |

### Pointer Type

**Phase 43 simple approach:**
```flow
let p: ptr = null              // Null pointer (0)
let addr = &variable           // Address-of operator (for FFI)
```

**Implementation:** `Value::Ptr(usize)` OR `Value::Float(addr as f64)`

**Recommendation:** Start with Value::Float for addresses (no new variant needed), add Value::Ptr in Phase 44 if needed.

---

## Implementation Phases

### Phase 43: Basic FFI (Parse Only)

**Goal:** extern blocks parse correctly, stored in AST, but not executed yet.

```rust
// AST addition
pub enum Statement {
    // ... existing variants
    ExternBlock {
        library: String,
        functions: Vec<ExternFn>,
        span: Span
    },
}

pub struct ExternFn {
    pub name: String,
    pub params: Vec<(String, CType)>,  // (name, type)
    pub return_type: CType,
}

pub enum CType {
    U32, U64, F32, F64,
    Ptr,      // void*
    String,   // const char*
}
```

**Parser changes:**
- Recognize `extern "lib" { ... }` blocks
- Parse function signatures
- Store in AST

**No execution yet.** Just parsing and AST storage.

### Phase 44: FFI Execution

**Goal:** Actually call extern functions.

```rust
// Runtime state
struct FfiLibrary {
    name: String,
    handle: libloading::Library,
    functions: HashMap<String, libloading::Symbol<...>>
}

// Execution
fn call_extern_fn(
    lib: &FfiLibrary,
    fn_name: &str,
    args: Vec<Value>
) -> Result<Value, CliError> {
    let symbol = lib.functions.get(fn_name)?;
    // Use libffi or manual calling convention
    // Convert Value args to C types
    // Call function pointer
    // Convert C return to Value
}
```

**Dependency:** `libloading` crate for dlopen OR use std::os directly.

### Phase 50-51: FFI in .flow

**Goal:** The FFI dispatch logic itself is .flow.

```flow
// stdlib/ffi.flow (meta-circular!)

fn call_extern(lib: string, fn_name: string, args: [Value]) -> Value {
    let handle = dlopen(lib)              // FFI primitive
    let symbol = dlsym(handle, fn_name)   // FFI primitive
    return ffi_call(symbol, args)         // FFI primitive (actual call)
}
```

Only `dlopen`, `dlsym`, `ffi_call` stay in Rust. Everything else is .flow.

---

## Minimal Implementation for Phase 43

Since FFI execution is complex and we want to proceed quickly, **Phase 43 focuses on bitwise operators and regex (both are self-contained, high value).**

**FFI can be Phase 44** (gives more time to design properly).

**Revised Phase 43 scope:**
1. ✅ Bitwise operators (<<, >>, &, |, ^) — Agent working
2. ✅ Regex operations — Agent working
3. ⏭️ Enum + match — Deferred to Phase 44 (complex)
4. ⏭️ extern FFI — Deferred to Phase 44 (needs careful design)

**This lets us deliver high-value features NOW while designing FFI properly.**

---

## Next Steps

Wait for agents to complete bitwise + regex, test integration, commit Phase 43 with those features. Design enum/match + FFI properly for Phase 44.
