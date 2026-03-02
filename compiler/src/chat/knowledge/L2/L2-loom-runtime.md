# runtime (L2)
loom/ops/runtime — Vulkan GPU runtime with dispatch chains.

## Loom Engine Architecture
Main Loom + Support Loom = Loom Engine.
- Main Loom: GPU compute only. loom_dispatch → loom_build → loom_launch.
- Support Loom: CPU↔GPU I/O. loom_boot, loom_write, loom_read, loom_present, loom_wait.
- N Main Looms → 1 Support Loom. Support boots first.
- Patterns: Implicit (CPU as Support), Conflated (1 VM), Explicit (2 VMs).

## Core Functions
loom_boot(bind, reg_size, globals) → vm_id
  Create a compute unit with register and globals buffers
loom_dispatch(vm, spv_path, params, wg) → 0.0
  Record a kernel dispatch into the unit's chain
loom_build(vm) → prog_id
  Compile dispatch chain into Vulkan command buffer
loom_run(prog) → 0.0
  Execute synchronously (blocks until complete)
loom_launch(prog) → 0.0
  Execute asynchronously (returns immediately)
loom_poll(prog) → 0.0|1.0
  Check if async execution completed
loom_wait(prog) → 0.0
  Block until async execution completes
loom_read(vm, bind, off, len) → array
  Read results from GPU memory
loom_write(vm, offset, data) → 0.0
  Upload data to globals buffer
loom_present(vm, total) → 0.0
  Download framebuffer and blit to window
loom_shutdown(vm) → 0.0
  Destroy VM and free GPU resources
loom_set_heap(vm, data) → 0.0
  Upload data to heap buffer (large transfers)
loom_free(prog) → 0.0
  Free a compiled program
loom_status(vm) → pace_us
  Query homeostasis pacing delay
loom_pace(vm, us) → 0.0
  Manually set pacing delay (0 = disable)
loom_prefetch(spv_path) → 0.0
  Background-load SPIR-V file for later dispatch

## Legacy Functions (stdlib wrappers)
rt_init() → map
rt_cleanup(rt) → int
rt_load_pipeline(rt, spv) → map
rt_create_buffer(rt, size) → map
rt_upload(rt, buf, data) → int
rt_download(rt, buf) → array
rt_chain_begin(rt) → map
rt_chain_dispatch(chain, pipe, x) → map
rt_chain_end(chain) → map
rt_chain_submit_wait(rt, chain) → int
