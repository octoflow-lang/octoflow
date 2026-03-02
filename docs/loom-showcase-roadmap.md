# Loom Engine — Showcase Roadmap

> Six GPU showcases proving the Loom Engine across physics, adaptive computing,
> fluid dynamics, computational chemistry, quantitative finance, and cryptography.

**Hardware target:** NVIDIA GeForce GTX 1660 SUPER (1408 cores, 6GB GDDR6, 1530/1785 MHz)
**CPU:** AMD Ryzen 5 4600G (6 cores / 12 threads, 3.7 GHz)

---

## SC-1: N-Body 3D Ray Traced (Physics)

**Status:** COMPLETE — shipped in Phase 3O+ showcase.

**Summary:**
- 4096 particles with spatial hash grid for O(N) gravity
- Real-time ray tracing (sphere intersection, Blinn-Phong shading)
- 2 Main Looms (compute + render) + Support Loom
- Target: 120 FPS at 800×600
- Kernels: `spatial_hash`, `spatial_count`, `spatial_scatter`, `spatial_gravity`, `nbody_raytrace`

---

## SC-1B: N-Body Quad-View (Adaptive Loom Showcase)

**Status:** PLANNED — requires Phase 4A (Mailbox) to ship first.

**Depends on:** SC-1 (all kernels), Phase 4A-03 (Mailbox builtins)

### What It Proves

The Adaptive Loom Engine can run **multiple parallel render pipelines** fed by a
single physics simulation, with **cross-loom communication** via mailbox. Visually:
four synchronized camera views of the same 3D N-Body simulation on a 2×2 grid —
like a mission control wall of monitors, all driven by one GPU.

This is the definitive showcase for Phase 4 because it exercises:
- **Mailbox:** Physics loom broadcasts particle state to 4 render looms
- **Multi-Stack (Parallel):** 4 render stacks running simultaneously
- **On-Demand Looms:** 6 VMs spawned (1 physics + 4 render + 1 compositor)
- **Resource Budget:** 6 VMs within VRAM budget, composited to single window
- **100% Kernel Reuse:** Every GPU kernel comes from SC-1, zero new shaders

### Loom Architecture

```
┌───────────────────────────────────────────────────────────────┐
│                  N-BODY QUAD-VIEW                              │
│                                                                │
│  SUPPORT LOOM (CPU orchestrator)                              │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │  1. Dispatch physics step on Physics Loom                │ │
│  │  2. Mailbox broadcast: particle state → 4 render looms   │ │
│  │  3. Dispatch ray trace on all 4 render looms (parallel)  │ │
│  │  4. Composite 4 quadrants → full framebuffer             │ │
│  │  5. Present to window                                    │ │
│  │  6. Update orbit camera angle each frame                 │ │
│  └──────────────────────────────────────────────────────────┘ │
│                                                                │
│  ┌──────────────┐        ┌──────────────────────────────────┐ │
│  │ PHYSICS LOOM │        │          MAILBOX                  │ │
│  │ (Main Loom 0)│───────►│  Ring buffer: 4 slots            │ │
│  │              │ send   │  slot_size = PARTICLE_COUNT × 4  │ │
│  │ spatial_hash │        │  (x, y, z, radius per particle)  │ │
│  │ spatial_count│        └──────┬───┬───┬───┬───────────────┘ │
│  │ spatial_scat │               │   │   │   │  recv ×4        │
│  │ spatial_grav │               ▼   ▼   ▼   ▼                │
│  │              │        ┌─────┐┌─────┐┌─────┐┌─────┐       │
│  │ 4096 particles│       │ R-0 ││ R-1 ││ R-2 ││ R-3 │       │
│  └──────────────┘        │Front││ Top ││Side ││Orbit│       │
│                          │     ││     ││     ││     │       │
│                          │cam: ││cam: ││cam: ││cam: │       │
│                          │z=10 ││y=10 ││x=10 ││spin │       │
│                          │     ││     ││     ││     │       │
│                          │640  ││640  ││640  ││640  │       │
│                          │×360 ││×360 ││×360 ││×360 │       │
│                          └──┬──┘└──┬──┘└──┬──┘└──┬──┘       │
│                             │      │      │      │           │
│                             ▼      ▼      ▼      ▼           │
│                    ┌──────────────────────────────────┐       │
│                    │      COMPOSITOR (Main Loom 5)     │      │
│                    │                                    │      │
│                    │  ┌──────────┬──────────┐          │      │
│                    │  │  R-0 fb  │  R-1 fb  │          │      │
│                    │  │  front   │  top     │          │      │
│                    │  ├──────────┼──────────┤          │      │
│                    │  │  R-2 fb  │  R-3 fb  │          │      │
│                    │  │  side    │  orbit   │          │      │
│                    │  └──────────┴──────────┘          │      │
│                    │       1280 × 720 window           │      │
│                    │       vm_present(compositor)       │      │
│                    └──────────────────────────────────┘       │
│                                                                │
└───────────────────────────────────────────────────────────────┘
```

### Camera Setup (4 Viewpoints)

| Viewport | Position | Grid | Camera Push Constants |
|----------|----------|------|----------------------|
| **R-0: Front** | Top-left | (0,0) | eye=(0, 0, 10), target=(0, 0, 0), up=(0, 1, 0) |
| **R-1: Top-Down** | Top-right | (1,0) | eye=(0, 10, 0), target=(0, 0, 0), up=(0, 0, -1) |
| **R-2: Side** | Bottom-left | (0,1) | eye=(10, 0, 0), target=(0, 0, 0), up=(0, 1, 0) |
| **R-3: Orbit** | Bottom-right | (1,1) | eye=(10·cos(t), 3, 10·sin(t)), target=(0, 0, 0), up=(0, 1, 0) |

The orbit camera rotates around the Y axis, creating a continuously changing viewpoint.
`t` increments by 0.02 per frame (~3.6° per frame, full orbit every ~100 frames).

### Buffer Layout

**Per render loom (640×360 viewport):**
```
Framebuffer:  640 × 360 × 3 (R, G, B planar) = 691,200 floats
Particles:    4096 × 4 (x, y, z, radius)      = 16,384 floats
Total globals: ~710,000 floats (~2.7 MB)
```

**Compositor loom (1280×720 full frame):**
```
Framebuffer:  1280 × 720 × 3 = 2,764,800 floats (~10.6 MB)
```

**Total VRAM for all 6 VMs:**
```
Physics loom:     ~2 MB (particles + spatial hash buffers)
4 × Render looms: ~2.7 MB × 4 = ~10.8 MB
Compositor loom:  ~10.6 MB
Mailbox:          ~0.1 MB (small ring buffer)
──────────────────────────────────────
Total:            ~23.5 MB  ← trivial on 6GB GPU
```

### Frame Pipeline (Per Frame)

```
Step 1: Physics (sequential)                        ~2.0 ms
  ├── spatial_hash.spv      (4096 particles)
  ├── spatial_count.spv     (32K cells)
  ├── prefix_sum            (CPU, 32K cells)
  ├── spatial_scatter.spv   (4096 particles)
  └── spatial_gravity.spv   (4096 particles, O(N))

Step 2: Mailbox broadcast                           ~0.5 ms
  ├── loom_mail_send(mb, physics, PART_OFF, PART_COUNT)  ×1
  └── loom_mail_recv(mb, render[i], 0)                   ×4

Step 3: Ray trace (4 viewports in parallel)         ~3.0 ms
  ├── Render 0: nbody_raytrace.spv (640×360, front cam)
  ├── Render 1: nbody_raytrace.spv (640×360, top cam)
  ├── Render 2: nbody_raytrace.spv (640×360, side cam)
  └── Render 3: nbody_raytrace.spv (640×360, orbit cam)
  (All 4 dispatched + built + launched simultaneously)

Step 4: Composite                                   ~0.5 ms
  ├── loom_copy(render0, fb_off, compositor, quad0_off, fb_size)
  ├── loom_copy(render1, fb_off, compositor, quad1_off, fb_size)
  ├── loom_copy(render2, fb_off, compositor, quad2_off, fb_size)
  └── loom_copy(render3, fb_off, compositor, quad3_off, fb_size)

Step 5: Present                                     ~1.0 ms
  └── vm_present(compositor)  (async, overlapped with next frame)

TOTAL PER FRAME: ~7.0 ms → ~142 FPS
```

### Composition Strategy

Two approaches for assembling the 4 viewports into one 1280×720 framebuffer:

**Option A: Row-by-row loom_copy (simple, no new kernel)**

OctoFlow uses planar RGB framebuffers (all R values, then all G, then all B).
For a 2×2 grid, each scanline of the full image is assembled from two half-scanlines:

```
For each row y in [0, 360):
  Full row [0..1280) = Render0 row [0..640) + Render1 row [0..640)

For each row y in [360, 720):
  Full row [0..1280) = Render2 row [0..360) + Render3 row [0..360)
```

With planar layout, this becomes 6 large copies per channel (R, G, B):
- Top-left block: contiguous in source, contiguous in dest (single copy per channel)
- Top-right block: offset in dest (single copy per channel)
- etc.

Total: 12 `loom_copy` calls (4 viewports × 3 channels). Each is a GPU-to-GPU DMA.

**Option B: Compositor kernel (faster, one dispatch)**

Emit a simple `compositor.spv` kernel that reads from 4 input bindings and writes
to one output binding, using `gid` to compute which quadrant each pixel belongs to:

```
fn emit_compositor(name, W, H, half_w, half_h)
  ir_new(name, 1)
  let body = ir_block("main")
  let gid = ir_gid(body)
  let px = ir_umod(body, gid, W)       // x = gid % 1280
  let py = ir_udiv(body, gid, W)       // y = gid / 1280
  // Determine quadrant and source binding
  let qx = ir_udiv(body, px, half_w)   // 0 or 1
  let qy = ir_udiv(body, py, half_h)   // 0 or 1
  let src_binding = ir_iadd(body, ir_imul(body, qy, 2), qx)  // 0,1,2,3
  let local_x = ir_umod(body, px, half_w)
  let local_y = ir_umod(body, py, half_h)
  let src_idx = ir_iadd(body, ir_imul(body, local_y, half_w), local_x)
  // Read from quadrant source, write to full framebuffer
  let val = ir_buf_load(body, src_binding, src_idx)
  ir_buf_store(body, 4, gid, val)       // binding 4 = output
  ir_write_spv(name + ".spv")
end
```

One dispatch, 1280×720 threads, reads from 4 source buffers, writes to 1 output.
Much faster than 12 separate copies.

**Recommendation:** Option B (compositor kernel). It's a trivial emitter (~20 lines)
and eliminates the complexity of row-by-row copy with planar layout offsets.

### .flow Code Sketch (~80 lines)

```
// === N-Body Quad-View: 4 cameras, 1 simulation, 6 looms ===

let W = 1280.0
let H = 720.0
let HW = 640.0
let HH = 360.0
let N = 4096.0
let PART_SIZE = N * 4.0

// Boot looms
let physics = loom_boot(1.0, PHYS_REGS, PHYS_GLOBALS)
let render = []
let ri = 0.0
while ri < 4.0
  push(render, loom_boot(1.0, 1.0, HW * HH * 3.0 + PART_SIZE))
  ri = ri + 1.0
end
let comp = loom_boot(1.0, 1.0, W * H * 3.0)

// Mailbox for particle broadcast
let mb = loom_mailbox(PART_SIZE, 4.0)

// Emit compositor kernel (or use pre-emitted)
emit_compositor("quad_comp", W, H, HW, HH)

// Camera definitions: [eye_x, eye_y, eye_z, target_x, target_y, target_z]
let cam_front = [0.0, 0.0, 10.0,  0.0, 0.0, 0.0]
let cam_top   = [0.0, 10.0, 0.0,  0.0, 0.0, 0.0]
let cam_side  = [10.0, 0.0, 0.0,  0.0, 0.0, 0.0]
let cam_orbit = [10.0, 3.0, 0.0,  0.0, 0.0, 0.0]
let cameras = [cam_front, cam_top, cam_side, cam_orbit]

// Initialize particles on physics loom (random sphere distribution)
// ... (same as SC-1 init)

let frame = 0.0
while frame < 100000.0
  // --- PHYSICS STEP ---
  loom_dispatch(physics, "spatial_hash.spv", pc_hash, WG_HASH)
  loom_dispatch(physics, "spatial_count.spv", pc_count, WG_COUNT)
  loom_build(physics)
  let phys_prog = loom_run(physics)

  // CPU prefix sum on cell counts (same as SC-1)
  // ...

  loom_dispatch(physics, "spatial_scatter.spv", pc_scatter, WG_SCATTER)
  loom_dispatch(physics, "spatial_gravity.spv", pc_gravity, WG_GRAVITY)
  loom_build(physics)
  let phys_prog2 = loom_run(physics)

  // --- MAILBOX BROADCAST ---
  loom_mail_send(mb, physics, PART_OFFSET, PART_SIZE)
  ri = 0.0
  while ri < 4.0
    loom_mail_recv(mb, render[ri], 0.0)
    ri = ri + 1.0
  end

  // --- PARALLEL RAY TRACE (4 viewports) ---
  // Update orbit camera
  let angle = frame * 0.02
  cameras[3][0] = 10.0 * cos(angle)
  cameras[3][2] = 10.0 * sin(angle)

  let render_progs = []
  ri = 0.0
  while ri < 4.0
    let pc_rt = [HW, HH, N]
    // Append camera params to push constants
    let ci = 0.0
    while ci < 6.0
      push(pc_rt, cameras[ri][ci])
      ci = ci + 1.0
    end
    loom_dispatch(render[ri], "nbody_raytrace.spv", pc_rt, RT_WG)
    loom_build(render[ri])
    push(render_progs, loom_launch(render[ri]))
    ri = ri + 1.0
  end

  // Wait for all 4 renders
  ri = 0.0
  while ri < 4.0
    loom_wait(render_progs[ri])
    ri = ri + 1.0
  end

  // --- COMPOSITE ---
  // Copy 4 framebuffers into compositor's globals
  ri = 0.0
  while ri < 4.0
    let fb_off = PART_SIZE  // framebuffer starts after particle data in render VM
    let comp_off = ri * HW * HH * 3.0  // each quadrant's region in compositor
    loom_copy(render[ri], fb_off, comp, comp_off, HW * HH * 3.0)
    ri = ri + 1.0
  end

  // Dispatch compositor kernel to assemble 2×2 grid
  loom_dispatch(comp, "quad_comp.spv", [W, H, HW, HH], ceil(W * H / 256.0))
  loom_build(comp)
  loom_run(comp)

  vm_present(comp)
  frame = frame + 1.0
end

// Cleanup
loom_mailbox_destroy(mb)
ri = 0.0
while ri < 4.0
  loom_shutdown(render[ri])
  ri = ri + 1.0
end
loom_shutdown(physics)
loom_shutdown(comp)
```

### Kernel Inventory (100% Reuse + 1 New)

| Kernel | Source | New? |
|--------|--------|------|
| `spatial_hash.spv` | `emit_spatial_hash.flow` (SC-1) | No |
| `spatial_count.spv` | `emit_spatial_count.flow` (SC-1) | No |
| `spatial_scatter.spv` | `emit_spatial_scatter.flow` (SC-1) | No |
| `spatial_gravity.spv` | `emit_spatial_gravity.flow` (SC-1) | No |
| `nbody_raytrace.spv` | `emit_nbody_raytrace.flow` (SC-1) | No |
| `quad_comp.spv` | `emit_compositor.flow` (new, ~20 lines) | **Yes** |

Only one new kernel emitter needed — the trivial compositor that maps `gid` to
quadrant and copies pixels.

### Implementation Plan

| # | Task | Owner | Depends On |
|---|------|-------|-----------|
| 1 | Emit `quad_comp.spv` compositor kernel | Dev 2 | None (pure IR emitter) |
| 2 | Write `nbody_quadview.flow` showcase | Dev 2 | Phase 4A-03 (Mailbox) |
| 3 | Test 4-viewport rendering | Dev 2 | Task 1 + 2 |
| 4 | Performance tuning + screenshot | Auditor | Task 3 |

**Estimated effort:** Small — all physics + rendering kernels exist. The showcase
is primarily .flow orchestration code (~80 lines) plus one trivial emitter.

### Success Criteria

- 4 viewports render simultaneously at ≥120 FPS
- Orbit camera rotates smoothly
- Particle positions identical across all 4 views (mailbox integrity)
- Total VRAM usage < 50 MB (well within 6GB budget)
- Visually compelling: 4096 ray-traced particles from 4 angles

---

## SC-2: Interactive Fluid Simulation (Lattice Boltzmann)

### What It Proves

The Loom Engine handles **continuous physics simulation** with real-time visualization
and interactive user input — all with zero thread management in .flow.

### Loom Architecture

```
┌──────────────────────────────────────────────────────────┐
│                  FLUID SIMULATION                         │
│                                                           │
│  SUPPORT LOOM (CPU)                                       │
│  ┌─────────────────────────────────────────────────────┐ │
│  │  Mouse input → inject velocity/density into grid    │ │
│  │  Transfer density field from sim → render VM        │ │
│  │  Present framebuffer to window                      │ │
│  │  Parameter tuning (viscosity, speed via keyboard)   │ │
│  └─────────────────────────┬───────────────────────────┘ │
│                             │                             │
│              ┌──────────────┼──────────────┐              │
│              ▼                             ▼              │
│  ┌───────────────────┐         ┌───────────────────┐     │
│  │  MAIN LOOM 1      │         │  MAIN LOOM 2      │     │
│  │  (Simulation)     │         │  (Rendering)      │     │
│  │                   │         │                   │     │
│  │  Collision step   │         │  Density → color  │     │
│  │  Streaming step   │         │  Velocity arrows  │     │
│  │  Boundary conds   │         │  Vorticity color  │     │
│  │                   │         │                   │     │
│  │  512×256 D2Q9     │         │  512×256 RGB      │     │
│  └───────────────────┘         └───────────────────┘     │
│                                                           │
└──────────────────────────────────────────────────────────┘
```

### Algorithm: Lattice Boltzmann Method (D2Q9)

The LBM simulates fluid by tracking **distribution functions** on a 2D grid.
Each cell has 9 distribution values (one per lattice velocity direction).
Two steps alternate every frame:

**1. Collision (relaxation toward equilibrium):**
```
For each cell (i,j):
  ρ = sum of all 9 distributions            // density
  u = momentum-weighted average / ρ          // velocity
  f_eq[k] = ρ × w[k] × (1 + 3(e[k]·u) + 4.5(e[k]·u)² - 1.5|u|²)
  f[k] = f[k] - (f[k] - f_eq[k]) / τ       // BGK relaxation
```
Pure local computation — each cell is independent. Perfectly parallel.

**2. Streaming (propagate to neighbors):**
```
For each cell (i,j):
  For each direction k:
    f_new[i + ex[k], j + ey[k]][k] = f[i,j][k]
```
Neighbor write — requires double-buffered grids to avoid data races.

**3. Boundary conditions:**
```
For wall cells:    bounce-back (reverse distributions)
For inlet cells:   fixed velocity (Zou-He)
For outlet cells:  zero-gradient (copy from interior)
```

### Kernel Emitters Needed

| # | Emitter | File | Algorithm | Workgroup |
|---|---------|------|-----------|-----------|
| K-1 | `emit_lbm_collide.flow` | `stdlib/loom/emit/science/` | BGK collision + mouse force injection | 256 |
| K-2 | `emit_lbm_stream.flow` | `stdlib/loom/emit/science/` | Distribution propagation (double-buffer swap) | 256 |
| K-3 | `emit_lbm_render.flow` | `stdlib/loom/emit/science/` | Density/velocity → RGB color mapping | 256 |

### Buffer Layout (Simulation VM)

```
Grid: 512 × 256 = 131,072 cells
D2Q9: 9 distributions per cell

Binding 0 (globals):
  [0 .. 131072×9)              = f_current[W×H×9]       (1,179,648 floats)
  [131072×9 .. 131072×18)      = f_next[W×H×9]          (1,179,648 floats)
  [131072×18 .. 131072×19)     = boundary_flags[W×H]    (131,072 floats)
  [131072×19 .. 131072×19+4)   = mouse_force[4]         (x, y, fx, fy)

Total: ~2,490,372 floats ≈ 9.5 MB
```

### Push Constants

```
Collision: [W, H, tau, mouse_x, mouse_y, force_strength]
Stream:    [W, H]
Render:    [W, H, color_mode]  // 0=density, 1=velocity, 2=vorticity
```

### IR Ops Required

All available in the IR builder:
- `ir_buf_load_f` / `ir_buf_store_f` — grid reads/writes
- `ir_fmul`, `ir_fadd`, `ir_fsub`, `ir_fdiv` — BGK arithmetic
- `ir_load_gid` — thread index → cell coordinate
- `ir_ftou`, `ir_utof` — index arithmetic
- `ir_folt`, `ir_fogt`, `ir_select` — boundary condition checks

No atomics needed. No shared memory needed. LBM is simpler than N-Body.

### .flow Sketch (~70 lines)

```flow
use "stdlib/loom/emit/science/emit_lbm_collide"
use "stdlib/loom/emit/science/emit_lbm_stream"
use "stdlib/loom/emit/science/emit_lbm_render"

let W = 512.0
let H = 256.0
let TAU = 0.6       // relaxation time (viscosity control)
let CELLS = W * H

// Emit kernels
emit_lbm_collide("lbm_collide.spv")
emit_lbm_stream("lbm_stream.spv")
emit_lbm_render("lbm_render.spv")
loom_prefetch("lbm_collide.spv")
loom_prefetch("lbm_stream.spv")
loom_prefetch("lbm_render.spv")

// Boot VMs
let sim_vm    = loom_boot(1.0, 4.0, CELLS * 19.0 + 4.0)    // 2× grid + boundary + mouse
let render_vm = loom_boot(1.0, 4.0, W * H * 3.0 + CELLS)   // RGB framebuffer + density

// Initialize: uniform density, zero velocity
let mut init_f = []
let mut ci = 0.0
while ci < CELLS
  // Equilibrium at rest: f[k] = w[k] × ρ (ρ=1.0)
  // w = [4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36]
  push(init_f, 4.0 / 9.0)    // center
  push(init_f, 1.0 / 9.0)    // E
  push(init_f, 1.0 / 9.0)    // N
  push(init_f, 1.0 / 9.0)    // W
  push(init_f, 1.0 / 9.0)    // S
  push(init_f, 1.0 / 36.0)   // NE
  push(init_f, 1.0 / 36.0)   // NW
  push(init_f, 1.0 / 36.0)   // SW
  push(init_f, 1.0 / 36.0)   // SE
  ci = ci + 1.0
end
loom_write(sim_vm, 0.0, init_f)

// Set boundary flags (top/bottom walls)
let mut boundaries = []
// ... mark wall cells ...
loom_write(sim_vm, CELLS * 18.0, boundaries)

// Window
let _w = ui_window_open(W, H, "Fluid Simulation — Loom Engine")

// Frame loop
let mut t0 = now_ms()
let mut frame = 0.0

while ui_poll_events() >= 0.0
  // Mouse interaction: inject force at cursor
  let mx = gui_mouse_x()
  let my = gui_mouse_y()
  let mb = gui_mouse_button()
  if mb > 0.5
    loom_write(sim_vm, CELLS * 19.0, [mx, my, 0.1, 0.0])
  else
    loom_write(sim_vm, CELLS * 19.0, [0.0, 0.0, 0.0, 0.0])
  end

  // Collision step (pure local — embarrassingly parallel)
  loom_dispatch(sim_vm, "lbm_collide.spv", [W, H, TAU, mx, my, 0.1], ceil(CELLS / 256.0))

  // Streaming step (neighbor propagation, double-buffer swap)
  loom_dispatch(sim_vm, "lbm_stream.spv", [W, H], ceil(CELLS / 256.0))

  let sim_prog = loom_build(sim_vm)
  loom_launch(sim_prog)
  loom_wait(sim_prog)

  // Render: density → color
  let density = loom_read(sim_vm, 0, 0, CELLS)
  loom_write(render_vm, W * H * 3.0, density)
  loom_dispatch(render_vm, "lbm_render.spv", [W, H, 0.0], ceil(CELLS / 256.0))
  let render_prog = loom_build(render_vm)
  loom_launch(render_prog)
  loom_wait(render_prog)

  loom_present(render_vm, W * H)
  loom_free(sim_prog)
  loom_free(render_prog)

  // FPS
  frame = frame + 1.0
  if frame % 60.0 == 0.0
    let fps = 60000.0 / (now_ms() - t0)
    print("FPS: {fps}")
    t0 = now_ms()
  end
end
```

### Performance Estimate (GTX 1660 SUPER)

| Step | Time | Notes |
|------|------|-------|
| Collision (131K cells × 9 directions) | ~0.3ms | Pure arithmetic, no memory contention |
| Streaming (131K cells × 9 directions) | ~0.5ms | Neighbor reads, double-buffered |
| Density readback (512KB) | ~0.1ms | Async present overlap |
| Render (131K pixels) | ~0.1ms | Simple color mapping |
| Present + CPU overhead | ~0.5ms | |
| **Total** | **~1.5ms** | **660 FPS theoretical, 60 FPS capped by vsync** |

Could run at 1024×512 grid and still hit 120 FPS.

### Demo Scenario

1. Start: still fluid, uniform blue
2. User clicks and drags → injects velocity → fluid swirls
3. Toggle color mode (keyboard): density (blue→red), velocity (arrows), vorticity (rainbow)
4. Add obstacles: right-click places wall cells → flow around them
5. Adjust viscosity: up/down keys change τ (thick honey → thin water)

**Visual impact:** Interactive, immediate, beautiful. The kind of demo that gets shared.

---

## SC-3: Molecular Dynamics Simulation

### What It Proves

The Loom Engine handles **multi-physics scientific simulation** with concurrent
force computation, integration, and real-time analysis — the kind of workload
that currently requires GROMACS (100K+ lines of C++/CUDA).

### Loom Architecture

```
┌──────────────────────────────────────────────────────────┐
│              MOLECULAR DYNAMICS                           │
│                                                           │
│  SUPPORT LOOM (CPU)                                       │
│  ┌─────────────────────────────────────────────────────┐ │
│  │  Load initial coordinates (PDB/XYZ format)          │ │
│  │  Transfer positions between force/integrator VMs    │ │
│  │  Checkpoint trajectory to disk (every N steps)      │ │
│  │  Transfer positions to render VM for visualization  │ │
│  │  Print energy, temperature, pressure metrics        │ │
│  └────────────────────┬────────────────────────────────┘ │
│                        │                                  │
│       ┌────────────────┼─────────────────┐                │
│       ▼                ▼                 ▼                │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐             │
│  │ MAIN 1   │   │ MAIN 2   │   │ MAIN 3   │             │
│  │ (Forces) │   │ (Integ.) │   │ (Render) │             │
│  │          │   │          │   │          │             │
│  │ LJ + Coul│   │ Velocity │   │ Sphere   │             │
│  │ Spatial  │   │ Verlet   │   │ ray trace│             │
│  │ hash grid│   │ Thermost.│   │ (reuse   │             │
│  │ O(N)     │   │ Barostat │   │  SC-1!)  │             │
│  └──────────┘   └──────────┘   └──────────┘             │
│                                                           │
└──────────────────────────────────────────────────────────┘
```

### Algorithm

**Force computation (Main Loom 1):**
Each atom interacts with neighbors via:

1. **Lennard-Jones potential** (van der Waals):
```
V_LJ(r) = 4ε × [(σ/r)¹² - (σ/r)⁶]
F_LJ(r) = 24ε/r × [2(σ/r)¹² - (σ/r)⁶] × r̂
```

2. **Coulomb potential** (electrostatics):
```
V_C(r) = q_i × q_j / (4πε₀ × r)
F_C(r) = q_i × q_j / (4πε₀ × r²) × r̂
```

Both use **spatial hash grid** (same algorithm as N-Body SC-1) for O(N) neighbor search
with a cutoff distance (typically 1.0-1.2 nm).

**Integration (Main Loom 2):**
Velocity Verlet algorithm:
```
v(t + dt/2) = v(t) + F(t)/(2m) × dt        // half-step velocity
x(t + dt)   = x(t) + v(t + dt/2) × dt      // full-step position
// ... recompute forces at new positions ...
v(t + dt)   = v(t + dt/2) + F(t+dt)/(2m) × dt  // complete velocity
```

Berendsen thermostat: scale velocities to maintain target temperature.

**Analysis/Render (Main Loom 3):**
Reuse the N-Body ray trace kernel from SC-1. Atoms rendered as spheres
colored by element type (C=gray, N=blue, O=red, H=white).

### Kernel Emitters Needed

| # | Emitter | Algorithm | Reuses |
|---|---------|-----------|--------|
| K-1 | `emit_lj_coulomb_force.flow` | LJ + Coulomb with spatial hash grid | SC-1 spatial hash kernels |
| K-2 | `emit_verlet_integrate.flow` | Velocity Verlet + Berendsen thermostat | — |
| K-3 | `emit_rdf.flow` | Radial distribution function (histogram) | — |
| K-4 | (reuse) `emit_nbody_raytrace.flow` | Sphere ray tracing | SC-1 render kernel |

**Key reuse:** SC-1's spatial hash grid (hash → count → prefix_sum → scatter) is
the SAME algorithm for neighbor search in molecular dynamics. Only the interaction
kernel changes (gravity → Lennard-Jones + Coulomb).

### Buffer Layout (Force VM)

```
N = 10,000 atoms, GRID = 32 (cutoff-based grid)

Offset          | Data                     | Size (floats)
0               | positions[N×4]           | 40,000      (x, y, z, charge)
N×4             | velocities[N×4]          | 40,000      (vx, vy, vz, mass)
N×8             | forces[N×4]              | 40,000      (fx, fy, fz, 0)
N×12            | params[N×2]              | 20,000      (epsilon, sigma per atom type)
N×14            | cell_ids[N]              | 10,000
N×15            | counts[GRID³]            | 32,768
N×15+GRID³      | offsets[GRID³+1]         | 32,769
...             | sorted arrays            | ~60,000

Total: ~275,537 floats ≈ 1.1 MB
```

### Performance Estimate (GTX 1660 SUPER)

| Step | N=10K atoms | Notes |
|------|-------------|-------|
| Spatial hash (4 kernels) | ~0.5ms | Reuse from SC-1 |
| Force computation (LJ+Coulomb) | ~2ms | ~100 neighbors/atom avg |
| Integration (Verlet) | ~0.1ms | Per-atom, embarrassingly parallel |
| Render (optional, every 100 steps) | ~3ms | Ray trace reuse from SC-1 |
| CPU transfer + checkpoint | ~0.2ms | |
| **Per step (no render)** | **~2.8ms** | **~350 steps/sec** |
| **Per step (with render)** | **~5.8ms** | **~170 steps/sec** |

At 1 femtosecond timestep, 350 steps/sec = 350 fs/sec of simulated time.
A 10 nanosecond simulation (10M steps) takes ~8 hours.
Competitive with GROMACS on a single GPU for small systems.

### Demo Scenario

1. Load a small protein (e.g., insulin, ~800 atoms) or a water box (10K molecules)
2. Start simulation — atoms jiggle, protein folds
3. Real-time 3D visualization (rotating sphere rendering, colored by element)
4. Print temperature, energy, pressure at each step
5. Save trajectory checkpoint every 1000 steps

**Impact:** "Protein folding in 120 lines of .flow" — vs GROMACS's 100K lines of C++/CUDA.

---

## SC-4: Monte Carlo Option Pricing (Quantitative Finance)

### What It Proves

The Loom Engine handles **embarrassingly parallel stochastic simulation** — the
bread and butter of quantitative finance — with multiple analysis stages running
concurrently on GPU.

### Loom Architecture

```
┌──────────────────────────────────────────────────────────┐
│              MONTE CARLO PRICING                          │
│                                                           │
│  SUPPORT LOOM (CPU)                                       │
│  ┌─────────────────────────────────────────────────────┐ │
│  │  Market data input (spot price, vol, rates)         │ │
│  │  Transfer final prices from paths → payoff VM       │ │
│  │  Transfer payoffs → risk VM                         │ │
│  │  Print results: price, VaR, Greeks, confidence      │ │
│  └────────────────────┬────────────────────────────────┘ │
│                        │                                  │
│       ┌────────────────┼─────────────────┐                │
│       ▼                ▼                 ▼                │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐             │
│  │ MAIN 1   │   │ MAIN 2   │   │ MAIN 3   │             │
│  │ (Paths)  │   │ (Payoff) │   │ (Risk)   │             │
│  │          │   │          │   │          │             │
│  │ 1M GBM   │   │ Option   │   │ Sort     │             │
│  │ random   │   │ exercise │   │ VaR/CVaR │             │
│  │ walks    │   │ logic    │   │ Greeks   │             │
│  │          │   │ per path │   │ CI       │             │
│  └──────────┘   └──────────┘   └──────────┘             │
│                                                           │
└──────────────────────────────────────────────────────────┘
```

### Algorithm

**Path generation (Main Loom 1):**
Geometric Brownian Motion — each thread generates one price path:
```
For each thread i (i = 0..PATHS-1):
  S[0] = S0                              // initial price
  seed = hash(i, GLOBAL_SEED)            // deterministic per-path RNG
  For t = 1..STEPS:
    Z = box_muller(xorshift64(seed))     // standard normal
    S[t] = S[t-1] × exp((μ - σ²/2)×dt + σ×√dt×Z)
  Store S[STEPS-1] as final_price[i]
```

**Payoff computation (Main Loom 2):**
Each thread computes payoff for one path:
```
European Call:  payoff[i] = max(final_price[i] - K, 0)
European Put:   payoff[i] = max(K - final_price[i], 0)
Asian Call:     payoff[i] = max(avg(S[0..STEPS]) - K, 0)
Barrier:        payoff[i] = max(S[T] - K, 0) × (max(S) < barrier ? 1 : 0)
```

**Risk analysis (Main Loom 3):**
GPU-parallel sorting + percentile extraction:
```
Sort payoffs (GPU bitonic sort)
VaR(99%)    = payoffs[0.01 × PATHS]
CVaR(99%)   = mean(payoffs[0 .. 0.01 × PATHS])
Fair price  = exp(-r×T) × mean(payoffs)
Std error   = std(payoffs) / sqrt(PATHS)

Greeks (finite differences — rerun with bumped parameters):
  Delta = (price(S+dS) - price(S-dS)) / (2×dS)
  Gamma = (price(S+dS) - 2×price(S) + price(S-dS)) / dS²
  Vega  = (price(σ+dσ) - price(σ-dσ)) / (2×dσ)
```

### Kernel Emitters Needed

| # | Emitter | Algorithm | Complexity |
|---|---------|-----------|------------|
| K-1 | `emit_gbm_paths.flow` | GBM random walk with Box-Muller + xorshift64 | Medium |
| K-2 | `emit_option_payoff.flow` | Payoff function (call/put/asian/barrier) | Simple |
| K-3 | `emit_gpu_sort.flow` | Bitonic sort for risk percentiles | Medium (or reuse existing `gpu_sort`) |

### Buffer Layout (Paths VM)

```
PATHS = 1,000,000    STEPS = 252 (trading days)

Option A (store only final prices):
  [0 .. PATHS)           = final_prices[1M]         (1,000,000 floats = 4 MB)
  [PATHS .. PATHS+8)     = params: S0, μ, σ, dt, seed, ...

Option B (store full paths for Asian/barrier):
  [0 .. PATHS×STEPS)     = all_prices[252M]          (252M floats = 1 GB — too large)
  → Solution: compute path statistics inline (running avg, running max)
              and store only summary per path

Practical layout:
  [0 .. PATHS)           = final_prices[1M]           (4 MB)
  [PATHS .. 2×PATHS)     = path_avg[1M]               (4 MB, for Asian options)
  [2×PATHS .. 3×PATHS)   = path_max[1M]               (4 MB, for barrier options)

Total: 3M floats = 12 MB
```

### Performance Estimate (GTX 1660 SUPER)

| Step | Time | Notes |
|------|------|-------|
| Path generation (1M × 252 steps) | ~50ms | ~252M RNG + exp operations |
| Payoff computation (1M paths) | ~0.5ms | Per-path, trivial arithmetic |
| Sort (1M floats, bitonic) | ~5ms | 20 passes × 1M comparisons |
| Risk metrics (percentiles + mean) | ~1ms | GPU reduction |
| **Total** | **~57ms** | **1M paths in under 100ms** |

For comparison: Python + NumPy on CPU: ~5 seconds. CUDA: ~50ms.
Loom Engine matches CUDA performance with 10x less code.

### .flow Sketch (~60 lines)

```flow
use "stdlib/loom/emit/finance/emit_gbm_paths"
use "stdlib/loom/emit/finance/emit_option_payoff"

let PATHS = 1000000.0
let STEPS = 252.0
let S0 = 100.0      // spot price
let K = 105.0        // strike
let R = 0.05         // risk-free rate
let SIGMA = 0.2      // volatility
let T = 1.0          // 1 year
let DT = T / STEPS

emit_gbm_paths("gbm_paths.spv")
emit_option_payoff("option_payoff.spv")

let paths_vm  = loom_boot(1.0, 4.0, PATHS * 3.0 + 10.0)   // final + avg + max + params
let payoff_vm = loom_boot(1.0, 4.0, PATHS + 10.0)          // payoffs
let risk_vm   = loom_boot(1.0, 4.0, PATHS + 100.0)         // sorted payoffs + metrics

// Generate 1M price paths
loom_write(paths_vm, PATHS * 3.0, [S0, R, SIGMA, DT, STEPS, 42.0])  // params + seed
loom_dispatch(paths_vm, "gbm_paths.spv",
  [PATHS, STEPS, S0, R, SIGMA, DT, 42.0], ceil(PATHS / 256.0))
let paths_prog = loom_build(paths_vm)
loom_launch(paths_prog)
loom_wait(paths_prog)

// Compute payoffs
let finals = loom_read(paths_vm, 0, 0, PATHS)
loom_write(payoff_vm, 0.0, finals)
loom_dispatch(payoff_vm, "option_payoff.spv",
  [PATHS, K, 0.0], ceil(PATHS / 256.0))   // 0.0 = call
let payoff_prog = loom_build(payoff_vm)
loom_launch(payoff_prog)
loom_wait(payoff_prog)

// Risk analysis
let payoffs = loom_read(payoff_vm, 0, 0, PATHS)
let sorted = sort(payoffs)
let fair_price = exp(0.0 - R * T) * arr_avg(payoffs)
let var_99 = sorted[floor(PATHS * 0.01)]
let std_err = arr_std(payoffs) / sqrt(PATHS)

print("=== Monte Carlo Option Pricing ===")
print("Paths: {PATHS:.0}  Steps: {STEPS:.0}")
print("Spot: {S0}  Strike: {K}  Vol: {SIGMA}  Rate: {R}")
print("Fair Price:      {fair_price}")
print("VaR (99%):       {var_99}")
print("Std Error:       {std_err}")
print("95% CI:          [{fair_price} ± {std_err}]")
```

### Demo Scenario

1. Run with default parameters → print price, VaR, confidence interval
2. Compare with Black-Scholes analytical solution → match within std error
3. Bump to 10M paths → still under 1 second
4. Price exotic options (Asian, barrier) → same code, different payoff kernel
5. Compute Greeks via finite differences (6 pricing runs with bumped params)

**Impact:** "A quant analyst describes 'price Asian call with 1M paths' — LLM generates
60 lines of .flow that runs in 100ms. No CUDA. No QuantLib. No vendor lock-in."

---

## SC-5: Zero-Knowledge Proof Generation (Cryptography)

### What It Proves

The Loom Engine handles **algebraic cryptography** — finite field arithmetic,
polynomial operations, and multi-scalar multiplication — all GPU-parallel.
Currently NVIDIA-locked (CUDA). Loom Engine makes it hardware-agnostic.

### Loom Architecture

```
┌──────────────────────────────────────────────────────────┐
│              ZK PROOF GENERATION                          │
│                                                           │
│  SUPPORT LOOM (CPU)                                       │
│  ┌─────────────────────────────────────────────────────┐ │
│  │  Load witness (private inputs)                      │ │
│  │  Load proving key (SRS / CRS)                       │ │
│  │  Serialize proof output                             │ │
│  │  Verification (CPU — fast, single-threaded)         │ │
│  └────────────────────┬────────────────────────────────┘ │
│                        │                                  │
│       ┌────────────────┼─────────────────┐                │
│       ▼                ▼                 ▼                │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐             │
│  │ MAIN 1   │   │ MAIN 2   │   │ MAIN 3   │             │
│  │ (NTT)    │   │ (MSM)    │   │ (Hash)   │             │
│  │          │   │          │   │          │             │
│  │ Number   │   │ Multi-   │   │ Poseidon │             │
│  │ Theoretic│   │ Scalar   │   │ hash     │             │
│  │ Transform│   │ Multiply │   │ chain    │             │
│  │ (poly    │   │ (Pippen- │   │ (Fiat-   │             │
│  │  eval)   │   │  ger's)  │   │  Shamir) │             │
│  └──────────┘   └──────────┘   └──────────┘             │
│                                                           │
└──────────────────────────────────────────────────────────┘
```

### Background: Why GPU for ZK?

ZK proof generation is dominated by three operations:
1. **NTT (Number Theoretic Transform):** Polynomial multiplication in a prime field.
   Same structure as FFT but over integers modulo a prime. Butterfly pattern.
2. **MSM (Multi-Scalar Multiplication):** Compute ΣaᵢGᵢ where aᵢ are scalars
   and Gᵢ are elliptic curve points. The bottleneck of most ZK systems.
3. **Hash chains (Poseidon/Pedersen):** Generate challenges via Fiat-Shamir heuristic.

All three are massively parallelizable. Current state:
- **Rapidsnark** (CPU): slow
- **cuZK / Icicle** (CUDA): fast but NVIDIA-locked
- **Loom Engine:** hardware-agnostic via Vulkan. Runs on NVIDIA, AMD, Intel.

### Key Challenge: Finite Field Arithmetic

All ZK operations happen in a **prime field** Fₚ where p is a large prime
(typically 254 bits for BN254 or 255 bits for BLS12-381).

GPU native arithmetic is 32-bit float or 32-bit integer. For 256-bit field elements:
- Represent each element as 8 × 32-bit limbs (uint32)
- Implement modular add, subtract, multiply, reduce in GPU kernels
- Montgomery multiplication for efficient modular reduction

**IR builder support:**
- `ir_iadd`, `ir_imul` — 32-bit integer arithmetic (available)
- `ir_buf_load_u`, `ir_buf_store_u` — unsigned integer buffer access (available)
- `ir_carry` — carry propagation (may need to implement via add + compare)
- Shared memory for limb-parallel operations (available via `ir_shared_load/store`)

### Kernel Emitters Needed

| # | Emitter | Algorithm | Complexity |
|---|---------|-----------|------------|
| K-1 | `emit_field_arith.flow` | Montgomery multiply, add, sub in Fₚ (8-limb) | High |
| K-2 | `emit_ntt.flow` | Butterfly NTT over Fₚ (radix-2, GPU parallel) | High |
| K-3 | `emit_msm_pippenger.flow` | Pippenger's MSM with bucket accumulation | High |
| K-4 | `emit_poseidon_hash.flow` | Poseidon permutation over Fₚ | Medium |

### Buffer Layout (NTT VM)

```
Polynomial degree: n = 2²⁰ = 1,048,576 coefficients
Each coefficient: 8 × uint32 = 32 bytes (256-bit field element)

Binding 0:
  [0 .. n×8)             = poly_a[n×8]    (8,388,608 uint32 = 32 MB)
  [n×8 .. 2n×8)          = poly_b[n×8]    (32 MB)
  [2n×8 .. 2n×8 + n)     = twiddle[n×8]   (32 MB — precomputed roots of unity)

Total: ~96 MB (fits in 6 GB VRAM)
```

### Performance Estimate (GTX 1660 SUPER)

| Operation | n = 2²⁰ | Notes |
|-----------|---------|-------|
| NTT (forward) | ~100ms | 20 butterfly passes × 1M elements × 8-limb modmul |
| NTT (inverse) | ~100ms | Same as forward |
| Polynomial multiply (NTT + pointwise + iNTT) | ~250ms | 2 NTTs + 1M modmuls |
| MSM (Pippenger, 1M points, 256-bit scalars) | ~500ms | 256 buckets × 1M adds |
| Poseidon hash (1K hashes) | ~5ms | Per-hash: 64 field multiplications |
| **Full proof (Groth16-style)** | **~2-5 seconds** | Multiple NTTs + MSMs + hashes |

For comparison:
- CPU (Rapidsnark): ~30-60 seconds
- CUDA (Icicle): ~1-3 seconds
- Loom Engine: ~2-5 seconds (competitive, hardware-agnostic)

### .flow Sketch (~80 lines)

```flow
use "stdlib/loom/emit/crypto/emit_field_arith"
use "stdlib/loom/emit/crypto/emit_ntt"
use "stdlib/loom/emit/crypto/emit_poseidon"

let N = 1048576.0    // 2^20 polynomial degree
let LIMBS = 8.0      // 256-bit field elements as 8×32-bit

// Emit kernels
emit_field_arith("field_mul.spv")
emit_ntt("ntt_butterfly.spv")
emit_poseidon("poseidon.spv")

// Boot VMs
let ntt_vm  = loom_boot(1.0, 8.0, N * LIMBS * 3.0)     // poly_a + poly_b + twiddle
let msm_vm  = loom_boot(1.0, 8.0, N * LIMBS * 2.0)     // scalars + points
let hash_vm = loom_boot(1.0, 4.0, 100000.0)             // hash inputs/outputs

// Load proving key (Support Loom)
let twiddle_factors = compute_roots_of_unity(N)
loom_write(ntt_vm, N * LIMBS * 2.0, twiddle_factors)

// Load witness polynomial
let witness = load_witness("circuit.witness")
loom_write(ntt_vm, 0.0, witness)

// NTT: polynomial evaluation (20 butterfly passes)
let mut pass = 0.0
while pass < 20.0   // log2(N) = 20
  loom_dispatch(ntt_vm, "ntt_butterfly.spv",
    [N, pass, LIMBS], ceil(N / 256.0))
  pass = pass + 1.0
end
let ntt_prog = loom_build(ntt_vm)
loom_launch(ntt_prog)
loom_wait(ntt_prog)

// MSM: commitment computation
// ... dispatch Pippenger's algorithm ...

// Poseidon: Fiat-Shamir challenges
// ... dispatch hash chain ...

// Serialize proof
let proof_data = loom_read(ntt_vm, 0, 0, PROOF_SIZE)
write_file("proof.bin", proof_data)

let t = gpu_timer_end()
print("Proof generated in {t}ms")
print("Proof size: {PROOF_SIZE} bytes")
```

### Demo Scenario

1. Generate a proof for a simple circuit (e.g., "I know x such that SHA256(x) = y")
2. Print timing: NTT time, MSM time, hash time, total
3. Verify the proof (CPU, fast) — prints "VALID"
4. Compare with CPU baseline (Rapidsnark) — show 10-20x speedup
5. Run on AMD GPU → same code, same result (Vulkan portable)

**Impact:** "Hardware-agnostic ZK proof generation. Same .flow runs on NVIDIA, AMD,
Intel. No CUDA lock-in. The Web3 community has been asking for this."

### Why This Is Hard (Honest Assessment)

ZK is the most complex showcase. The 256-bit finite field arithmetic kernel
is non-trivial — Montgomery multiplication with carry propagation across 8 limbs
requires careful SPIR-V generation. This is the kind of work that tests whether
the IR builder can handle "real" cryptographic computation.

**Risk:** The IR builder may need extensions for 32-bit unsigned integer carry chains.
If `ir_iadd` doesn't expose carry flags, we may need a multi-instruction pattern
(add → compare → select carry → add to next limb).

**Mitigation:** Start with a smaller field (e.g., 128-bit = 4 limbs) to validate
the approach, then scale to 256-bit.

---

## Implementation Priority

```
SC-1: N-Body 3D             ← COMPLETE (shipped Phase 3O+)
    ↓ (reuses all SC-1 kernels + adds mailbox)
SC-1B: N-Body Quad-View     ← PLANNED (after Phase 4A mailbox ships)
    ↓
SC-2: Interactive Fluid      ← NEXT (simplest new showcase — no spatial hash needed)
    ↓ (reuses spatial hash from SC-1)
SC-3: Molecular Dynamics     ← AFTER SC-2 (reuses SC-1 spatial hash + ray trace)
    ↓ (independent)
SC-4: Monte Carlo Finance    ← PARALLEL with SC-3 (pure compute, no visualization)
    ↓ (independent, high complexity)
SC-5: ZK Proofs             ← LAST (needs finite field arithmetic — new IR patterns)
```

### Kernel Reuse Map

```
SC-1 (N-Body) ← COMPLETE
  ├── spatial hash kernels ──→ SC-1B (Quad-View) reuses 100%
  ├── ray trace kernel ──────→ SC-1B (Quad-View) reuses 100%
  ├── spatial hash kernels ──→ SC-3 (Molecular Dynamics) reuses neighbor search
  ├── ray trace kernel ──────→ SC-3 (Molecular Dynamics) reuses sphere rendering
  └── leapfrog integration ──→ SC-3 (Molecular Dynamics) similar Verlet integration

SC-1B (Quad-View)
  └── 1 new kernel: compositor (trivial ~20 lines, maps gid → quadrant)

SC-2 (Fluid/LBM)
  └── (standalone — no reuse needed, simplest new kernels)

SC-4 (Monte Carlo)
  └── (standalone — RNG + payoff + sort)

SC-5 (ZK Proofs)
  └── (standalone — finite field arithmetic is unique)
```

SC-1B reuses 100% of SC-1's kernels. SC-3 reuses SC-1's spatial hash + ray trace. SC-2, SC-4, SC-5 are independent.

---

## Summary

| Showcase | Kernels | Reuses From | New Code | Wow Factor |
|----------|---------|-------------|----------|------------|
| SC-1: N-Body | 7 | — | 7 emitters + showcase | Physics |
| SC-1B: Quad-View | 6 | SC-1 (5 kernels) | 1 compositor emitter + showcase | Adaptive Loom |
| SC-2: Fluid | 3 | — | 3 emitters + showcase | Visual, interactive |
| SC-3: MolDyn | 4 | SC-1 (3 kernels) | 1 new emitter + showcase | Science |
| SC-4: Monte Carlo | 3 | — | 3 emitters + showcase | Finance |
| SC-5: ZK Proofs | 4 | — | 4 emitters + showcase | Crypto, hardware-agnostic |

**Total new kernel emitters across all 6 showcases: ~19**
**Total new .flow showcases: 6 files, ~60-120 lines each**

> Six domains. Six showcases. One adaptive architecture.
> The Loom Engine makes GPU compute accessible to everyone.

---

*Loom Engine Showcase Roadmap — OctoFlow Documentation, March 2026*
