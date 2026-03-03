# OctoFlow Known Issues

Discovered during Rust deletion campaign and .flow stdlib development.
Tracked for audit and hardening.

## Language Limitations

### 1. push(arr, map_var) fails
- **Symptom**: `push(my_arr, my_map)` → "undefined scalar: my_map"
- **Root cause**: `push()` evaluates second arg via `eval_scalar`, but maps live in `hashmap_bindings` namespace
- **Impact**: Arrays-of-maps pattern is impossible
- **Workaround**: Restructure algorithms to avoid storing maps in arrays
- **Fix**: Extend push() to check hashmap_bindings when scalar lookup fails

### 2. Void user functions need explicit return
- **Symptom**: Bare `my_fn(args)` as ExprStmt → "scalar function did not return a value"
- **Root cause**: ExprStmt evaluates via eval_scalar which expects a Value return
- **Impact**: All .flow library functions must have `return 0.0` even if void
- **Workaround**: (a) Add `return 0.0` to all functions, (b) use `let _x = fn(args)` pattern
- **Fix**: Make ExprStmt tolerate missing return value (discard result)

### 3. Module use auto-appends .flow
- **Symptom**: `use "csv.flow"` looks for `csv.flow.flow`
- **Root cause**: Module loader always appends `.flow` to the argument
- **Impact**: Unintuitive for library authors
- **Workaround**: Use `use "csv"` not `use "csv.flow"`
- **Fix**: Detect and strip `.flow` suffix, or emit helpful error message

### 4. Builtin name collisions
- **Symptom**: .flow module function named `json_parse_array` collides with native builtin
- **Root cause**: Native builtins are checked before user functions in eval dispatch
- **Impact**: .flow modules can accidentally shadow builtins with confusing errors
- **Known collisions**: `normalize()` (native L2 unit vector shadows transform.flow min-max), `json_parse_array`
- **Workaround**: Rename .flow functions to avoid builtin names (e.g., `minmax_normalize`)
- **Fix**: Preflight should warn when user fn name matches a native builtin

### 5. No modulo operator
- **Symptom**: `x % y` is a syntax error
- **Impact**: Common math pattern unavailable
- **Workaround**: `x - floor(x / y) * y`

### 6. No scientific notation
- **Symptom**: `6.674e-11` rejected by parser
- **Impact**: Scientific constants require verbose encoding
- **Workaround**: `6.674 * pow(10.0, -11.0)`

### 7. No empty string literals
- **Symptom**: `""` rejected by parser
- **Impact**: Cannot initialize empty strings directly
- **Workaround**: Use `" "` (single space) as empty-ish string

### 8. No arithmetic in print interpolation
- **Symptom**: `print("Total: {a + b}")` fails
- **Impact**: Must precompute values before printing
- **Workaround**: `let total = a + b` then `print("Total: {total}")`

### 9. No inline array literals as function args
- **Symptom**: `foo([1.0, 2.0])` fails
- **Impact**: Must extract arrays to variables before passing
- **Workaround**: `let a = [1.0, 2.0]` then `foo(a)`

### 10. No nested GPU calls
- **Symptom**: `gpu_clamp(gpu_add(a, b), 0, 255)` fails
- **Impact**: GPU operations must be flat/sequential
- **Workaround**: `let tmp = gpu_add(a, b)` then `let r = gpu_clamp(tmp, 0, 255)`

### 11. NaN does not survive array storage
- **Symptom**: `let mut a = [0.0]` then `a[0] = 0.0 / 0.0` — reading `a[0]` gives 0.0, not NaN
- **Root cause**: Array element assignment or read-back sanitizes NaN values
- **Impact**: Cannot store NaN sentinel values in arrays; `debug_assert_no_nan` only useful for GPU buffer downloads
- **Workaround**: Use a sentinel float (e.g., -999.0) instead of NaN
- **Note**: Scalar NaN works correctly (`let x = 0.0 / 0.0` then `x != x` is true)

### 12. Shallow recursion limit (~20 depth)
- **Symptom**: Stack overflow (STATUS_STACK_OVERFLOW) at recursion depth ~50
- **Root cause**: Each fn call creates eval_scalar→execute_user_fn→execute_loop_body stack frames (~20KB each)
- **Impact**: Recursive algorithms limited to ~20 depth on default 1MB stack
- **Workaround**: Use iterative algorithms with while loops, or promote hot recursive functions to Rust builtins
- **Fix**: Increase default thread stack size, or implement tail-call optimization

### 13. ir.flow scalar parameters fail at 3+ module nesting depth
- **Status**: PARTIALLY resolved. Direct module calls work. Deep nesting through wrapper modules fails.
- **Works**: test_file → `use "emit_matmul_tiled"` → emit_matmul_tiled calls ir.flow → OK (2 levels)
- **Fails**: test_file → `use "silu"` → silu.flow's gpu_silu() calls emit_silu() which calls ir.flow → "undefined scalar: a1" (3+ levels through imported wrapper function)
- **Root cause**: When a function defined in an imported module calls a function from a transitively-imported module, scalar parameters don't bind correctly at 3+ levels of execute_user_fn nesting.
- **Workaround**: Separate emit (build step) from dispatch. Call emit functions from top-level, not from within wrapper functions. See `build_kernels.flow` pattern.
- **Fix**: Would require changes to execute_user_fn parameter binding in compiler.rs.

## Runtime Issues

### Function return truncation
- **Symptom**: gpu_variance/gpu_std return truncated values when called as functions
- **Root cause**: Function return through RETURNED_ARRAY may not preserve precision
- **Workaround**: Inline the reduction logic (no function calls)

### Module fn array return
- **Symptom**: Functions in imported modules returning arrays via `return arr_name` may not propagate
- **Root cause**: RETURNED_ARRAY thread-local side-channel scope issues
- **Workaround**: Use module-level arrays for output and copy-back semantics

### Transient GPU test failures
- **Symptom**: Occasional single GPU test failure that passes on re-run
- **Root cause**: Likely Vulkan driver timing / fence race condition
- **Impact**: CI flakiness
- **Workaround**: Re-run failed GPU tests once before declaring failure
