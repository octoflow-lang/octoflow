# Self-Hosted Compilation Status

**Current**: v1.17 — Self-hosted loader working, eval.flow has bugs
**Goal**: 99% self-hosted (< 500 lines Rust)

---

## What Works

✅ **loader.rs** (165 lines) — Invokes eval.flow successfully
✅ **eval.flow execution** — Can run and produce output
✅ **Simple programs** — print statements, basic let bindings work

## What Doesn't Work

❌ **Arithmetic**: a+b, a*b, a/b return wrong values
❌ **Loops**: Fibonacci computation returns 0
❌ **Functions**: Recursive factorial returns 0
❌ **Module imports**: Path resolution embeds quotes in paths

## Root Cause

**eval.flow is a meta-interpreter** (Stage 6 verified: eval.flow interprets eval.flow).
It was built to prove self-hosting is POSSIBLE, not to replace the full Rust compiler.

**Missing coverage in eval.flow**:
- Full arithmetic evaluation in all contexts
- Function call parameter passing
- Complex control flow
- Module import path resolution
- Many builtin functions

**The gap**: eval.flow ~= 60-70% language coverage. Full compiler ~= 100%.

---

## The Pragmatic Path

### Current Reality Check

**Can we delete compiler.rs now?** NO
- eval.flow has arithmetic bugs
- eval.flow can't handle module imports
- eval.flow missing many builtins
- Would break all existing programs

**What CAN we delete?** Nothing yet.

**What DO we need?** Fix eval.flow to 100% coverage OR build new .flow compiler.

---

## Two Paths Forward

### Path A: Fix eval.flow (Incremental)
**Time**: 1-2 weeks
**Approach**: Debug and extend eval.flow to handle all language features
**Pros**: Builds on existing 22,128 lines
**Cons**: eval.flow is complex meta-interpreter code, hard to debug

### Path B: New .flow Compiler (Clean Slate)
**Time**: 2-4 weeks  
**Approach**: Use existing .flow components:
- lexer.flow (exists)
- parser.flow (exists)
- preflight.flow (exists)
- ir.flow (exists)
- compiler_main.flow (new, 500 lines: ties them together)

**Pros**: Clean architecture, leverages proven components
**Cons**: Need to write compiler_main.flow

---

## Recommendation

**Path B is faster and cleaner.**

We have ALL the pieces:
1. lexer.flow — tokenizes .flow source ✅
2. parser.flow — builds AST ✅
3. preflight.flow — type checking ✅
4. ir.flow — SPIR-V emission ✅
5. Execution: Use Rust runtime for now (can move to .flow later)

**Just need**: compiler_main.flow (~500 lines) that calls these in sequence.

```flow
// compiler_main.flow
use "lexer"
use "parser"
use "preflight"
use "ir"

// Read source
let source = read_file(env("FLOW_INPUT"))

// Compile
let tokens = lex(source)
let ast = parse(tokens)
let ok = preflight_check(ast)
if not ok: exit(1)

// Execute (for now: call Rust runtime)
execute_ast(ast)
```

Then delete compiler.rs.

---

## Quick Win Available

**Don't wait for full self-hosting to delete Rust.**

Some modules are ALREADY redundant:
- lint.flow EXISTS (25 KB) → lint.rs (1,168 lines) can be deleted NOW
- preflight.flow EXISTS (25 KB) → preflight.rs (3,242 lines) can be deleted NOW

Just need to wire them up in the build.

---

## Revised Strategy

**Week 1**: Delete lint.rs + preflight.rs (4,410 lines eliminated)
**Week 2**: Write compiler_main.flow (500 lines)
**Week 3**: Test parity, delete compiler.rs (13,547 lines eliminated)
**Week 4**: Codecs to .flow (image_io.rs can wait)

**Result**: loader.rs (165) + vk_sys (150) + syscalls (100) + I/O (2,000) ≈ 2,500 lines Rust
**Then**: Move I/O to .flow → < 500 lines final

---

## Current Blocker

eval.flow has bugs that prevent it from being the primary compiler.
Options:
1. Debug eval.flow (hard, 22K lines of meta-interpreter)
2. Use component .flow files + new compiler_main.flow (easier)
3. Fix just module resolution, accept limited parity for now

**Recommended**: Option 2 (component approach).

---

**Status**: Self-hosted STRUCTURE proven. Full PARITY blocked on eval.flow bugs.
**Path**: Use component .flow files, write compiler_main.flow, delete Rust.
