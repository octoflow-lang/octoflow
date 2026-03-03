# Path to 100% Self-Hosted

**Current**: 45.7% self-hosted (21,940 .flow compiler / 48,005 total)
**Goal**: 99% self-hosted (< 500 lines Rust for OS boundary only)

---

## Current State

**What's Already in .flow:**
- ✅ eval.flow (265 KB, 22,128 lines) — full compiler + interpreter
- ✅ preflight.flow (25 KB) — type checking
- ✅ ir.flow (40 KB) — SPIR-V IR builder with OpAtomicIAdd
- ✅ lexer.flow — tokenizer
- ✅ parser.flow — AST construction
- ✅ codegen.flow — code generation

**What's Still in Rust:**
- compiler.rs (13,547 lines) — DUPLICATE of eval.flow functionality
- preflight.rs (3,242 lines) — DUPLICATE of preflight.flow
- lint.rs (1,168 lines) — needs lint.flow
- image_io.rs (1,686 lines) — codecs (JPEG, PNG, GIF, H.264)
- repl.rs (905 lines) — can be .flow
- Other I/O (csv, http, net, json, octo, etc.) — ~3,500 lines

**Total Rust**: 26,065 lines
**Target**: < 500 lines (OS boundary only)

---

## The Quick Path (Use What Exists!)

### Step 1: Create Thin Loader (1-2 hours)
**File**: `loader.rs` (~200-300 lines)

```rust
// Minimal Rust bootstrap
fn main() {
    // 1. Read .flow source from argv[1]
    // 2. Call eval.flow to compile + run
    // 3. Exit with status code
}
```

**What it does**:
- Parse command-line args
- Load eval.flow
- Pass source file to eval.flow
- eval.flow handles: lex → parse → preflight → eval → codegen → execute
- Report errors/results

### Step 2: Test Self-Hosted Compilation
```bash
# Old way (Rust compiler)
octoflow run examples/hello.flow

# New way (eval.flow compiler)
octoflow-selfhosted run examples/hello.flow
# Internally: loader.rs calls eval.flow which compiles+runs hello.flow
```

### Step 3: Verify Parity
- Run full test suite through eval.flow compiler
- Compare results with Rust compiler
- Fix any discrepancies

### Step 4: Make It Default
- Rename: `octoflow` → `octoflow-rust` (legacy)
- Rename: `octoflow-selfhosted` → `octoflow` (new default)
- Rust compiler becomes fallback/debug tool

---

## What Moves to .flow (After Loader Works)

### High Priority (Needed for 100% self-hosted)
1. **REPL** — repl.rs → repl.flow (~900 lines)
2. **Lint** — lint.rs → lint.flow (~1,200 lines)
3. **Main CLI** — main.rs logic → cli.flow

### Medium Priority (Can Defer)
4. **Codecs** — image_io.rs → .flow implementations
   - GIF decoder: ~250 lines (pure .flow)
   - JPEG decoder: ~800 lines
   - PNG decoder: ~400 lines
   - H.264 decoder: ~2,000 lines

### Low Priority (OS Boundary, Keep in Rust)
5. **File I/O** — Already has .flow wrappers, Rust is just syscalls
6. **Network** — Socket syscalls, keep in Rust
7. **Vulkan** — FFI to vulkan-1.dll, keep bindings
8. **CSV/JSON** — Can move to .flow

---

## The < 500 Line Rust Target

**What stays in Rust** (OS boundary):
- loader.rs (~250 lines) — main(), argv parsing, eval.flow invocation
- vk_sys.rs (~150 lines) — Vulkan FFI bindings
- syscalls.rs (~100 lines) — file/network/process syscalls

**Everything else**: Pure .flow

---

## Quick Win Strategy

**Week 1**: Create loader.rs + test parity
**Week 2**: Move REPL to .flow
**Week 3**: Move lint to .flow  
**Week 4**: Polish + verify 99% self-hosted

**Result**: OctoFlow binary that's 99% .flow, < 500 lines Rust.

---

## Why This Works

**eval.flow already does everything**:
- ✅ Lexing
- ✅ Parsing
- ✅ Preflight
- ✅ Evaluation
- ✅ Codegen (ir.flow)
- ✅ SPIR-V emission

**Rust compiler.rs is redundant!** It's a bootstrap that's no longer needed.

**The insight**: We've been maintaining TWO compilers (Rust + .flow). 
Just use the .flow one and delete the Rust one.

---

**Next Immediate Step**: Create loader.rs and test self-hosted compilation.

