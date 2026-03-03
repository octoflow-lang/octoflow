# Honest Assessment: Self-Hosting Status

**Previous**: eval.flow couldn't do arithmetic (2+3=2, fib(10)=0)
**Current**: ALL FIXED. eval.flow handles arithmetic, recursion, operator precedence.

---

## Bugs Found and Fixed (2026-02-21)

### Bug 1: Let-declaration arithmetic unreachable
**Root cause**: Binary op handler for `let x = a + b` was nested inside the
builtin fn call branch (line 768's `elif rtt == "ident" && ... "("` block).
When the next token after an ident was `+` instead of `(`, the binary op
handler was never reached. Code fell through to "single value rhs" which
just assigned the first operand.

**Fix**: Added new elif branch at the correct nesting level with full
two-pass operator precedence (Pass 1: * and /, Pass 2: + and -).
Handles both ident and float operands, string concatenation detection.

### Bug 2: Flat environment clobbers recursive calls
**Root cause**: eval.flow uses flat `env_num`/`env_s` maps. When `factorial(4)`
calls `factorial(3)`, the callee's `n=3` overwrites the caller's `n=4`.
For fibonacci with two recursive calls, local variables `a`, `b`, `fa`, `fb`
were all corrupted.

**Fix**: Three-part scope isolation:
1. Pre-scan: Collect each function's local variable names (`fn_locals` map)
2. Before call: Save caller's values for ALL callee params + locals to a stack
3. After return: Restore saved values, SKIPPING the return destination variable

### Bug 3: Return value overwritten by restore
**Root cause**: After `fib(b)` returned 1 into `fb`, the restore logic
would overwrite `fb` with its pre-call value (0).

**Fix**: Skip the `ret_vname` variable during restore.

---

## Current Test Results

```
eval.flow meta-interpreter:
  Arithmetic:     6/6 PASS (add, sub, mul, div, precedence x2)
  Functions:      8/8 PASS (add, mul, factorial, fibonacci, precedence, chained)
  Comprehensive: 20/20 PASS (loops, break, continue, arrays, maps, strings, fns)
  Hardening:     15/15 PASS

Rust test suite:  1,058/1,058 PASS (0 failed, 2 ignored)
```

---

## Updated Status

- eval.flow arithmetic: FIXED
- Function calls: FIXED (simple + recursive)
- Operator precedence: FIXED (two-pass: */first, +-second)
- Recursive scope isolation: FIXED (save/restore stack)
- Self-hosting Stage 6: VERIFIED (still works)

**Status**: Self-hosting bugs RESOLVED. Path to Rust deletion is unblocked.
