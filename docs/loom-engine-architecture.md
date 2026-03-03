# Loom Engine Architecture — The Octopus Model

Every Loom Engine is an octopus.

One brain. Many arms. The brain manages. The arms compute. Arms never talk to
the brain — the brain watches them. Arms self-regulate. The brain adjusts
strategy. This is not a metaphor. This is the architecture.

---

## The Two-Tier Split

The Loom Engine separates **management** from **compute** into two tiers:

```
                ┌──────────────────────────┐
                │      SUPPORT LOOM        │
                │       (The Brain)        │
                │                          │
                │  • Lifecycle management  │
                │  • Data I/O             │
                │  • Topology selection    │
                │  • Resource budgets     │
                │  • Adaptation           │
                └─────────┬────────────────┘
                          │ polls (never receives messages)
                     ┌────┴────┐
                     │         │
                  ┌──▼──┐  ┌──▼──┐
                  │ ARM │  │ ARM │  ...
                  │  0  │  │  1  │
                  └─────┘  └─────┘
                  Main Looms (GPU VMs)
                  • Raw GPU compute
                  • Self-regulating
                  • Autonomous
```

**Support Loom (the brain)** runs as `.flow` orchestration on the CPU. It boots
GPU VMs, distributes data, selects dispatch topology, and monitors health. It
never runs GPU compute itself.

**Main Looms (the arms)** are GPU virtual machines. Each arm gets one
`vkQueueSubmit` and one fence per dispatch chain. Arms run at full GPU speed
with no CPU synchronization during compute. Arms don't send messages to the
brain — the brain reads their state by polling.

Arms can optionally talk to each other via mailbox (ring-buffer GPU IPC).

### Why This Split Matters

Traditional GPU frameworks require the CPU to orchestrate every kernel launch.
The Loom Engine inverts this: the CPU sets up the work, then gets out of the
way. The GPU runs autonomously. The CPU only intervenes when the brain's
adaptation loop decides to change strategy.

This means:
- **No per-kernel CPU overhead** — dispatch chains queue multiple kernels, submit once
- **No CPU bottleneck** — arms self-correct without waiting for CPU decisions
- **Linear scaling** — add more arms, get more throughput

---

## Three Topology Modes

The brain selects a dispatch topology at runtime based on workload:

### Parallel

Same kernel, different data. Each arm processes a chunk.

```
Data: [████████████████████████]
       ↓      ↓      ↓      ↓
      Arm 0  Arm 1  Arm 2  Arm 3
       ↓      ↓      ↓      ↓
      [████] [████] [████] [████]  → gather results
```

Best for: embarrassingly parallel work (map, filter, element-wise ops).

### Sequential

Pipeline. Output of Arm N feeds into Arm N+1. GPU-to-GPU transfer — no CPU
roundtrip between stages.

```
Data → [Arm 0] → [Arm 1] → [Arm 2] → Result
        stage 1   stage 2   stage 3
```

Best for: multi-stage pipelines (encode → transform → decode).

### Hierarchical

Tree reduction. Arms pair up, reduce, repeat. Final result lands on Arm 0.

```
[Arm 0] [Arm 1] [Arm 2] [Arm 3]
   ╲      ╱        ╲      ╱
  [Arm 0]          [Arm 2]
       ╲              ╱
            [Arm 0]  ← final result
```

Best for: reductions (sum, max, min across large datasets).

The brain chooses topology automatically based on the number of kernels and
data size, or you can set it explicitly.

---

## Self-Regulation

Arms self-tune. The brain adjusts strategy.

Each arm has a built-in regulation loop that runs **inside the GPU dispatch
chain** — no CPU roundtrip. The arm monitors its own numerical stability and
adjusts scale factors automatically. If values drift outside bounds, the arm
corrects before the next kernel in the chain.

The brain periodically polls all arms and makes higher-level decisions:
- If too many arms are unstable, the brain reduces the workload
- If arms are idle, the brain consolidates
- The brain can change an arm's precision tier on the fly

This creates a two-level feedback system:
- **Per-arm:** fast, GPU-autonomous, reacts in microseconds
- **Per-engine:** slower, CPU-driven, reacts in milliseconds

The result: GPU compute that stays numerically stable without programmer
intervention. No NaN explosions. No gradient overflow. The arms handle it.

---

## Multi-Precision

Different arms can work at different precision levels simultaneously.

One engine might run attention layers at high fidelity and feedforward layers
at lower precision. The brain tracks each arm's precision tier and automatically
selects the right decompression kernel before compute.

This means:
- **Quality where it matters** — critical operations keep full precision
- **Speed where it's safe** — bulk operations run at reduced precision
- **Runtime-adaptive** — precision changes without recompilation

The dispatch chain handles decompression transparently: decompress → compute →
regulate, all in one GPU submission.

---

## Weight Management

Active weights in VRAM. Compressed weights in standby. Cold weights on disk.

```
┌─────────────────────────────┐
│ HOT — GPU VRAM (full size)  │  ← Active layer. Kernel reads directly.
├─────────────────────────────┤
│ WARM — GPU VRAM (compressed)│  ← Recent layers. Decompress to use.
├─────────────────────────────┤
│ COLD — Disk (compressed)    │  ← Distant layers. Stream when needed.
└─────────────────────────────┘
```

A 24GB model on a 6GB GPU: only the active layer needs full VRAM. Previous
layers compress and stay in GPU memory at reduced footprint — faster than
reloading from disk. Distant layers live on disk and stream back when needed.

The brain manages promotion (warm → hot) and eviction (hot → warm → cold) across
all arms. This happens transparently during inference — the application just
dispatches kernels and the engine handles the rest.

---

## Swarm — A Colony of Octopuses

Multiple engines running simultaneously. Each engine is an independent octopus
with its own brain and arms.

```
┌───────────────┐  ┌───────────────┐  ┌───────────────┐
│  Engine 0     │  │  Engine 1     │  │  Engine 2     │
│  ┌─────────┐  │  │  ┌─────────┐  │  │  ┌─────────┐  │
│  │  Brain  │  │  │  │  Brain  │  │  │  │  Brain  │  │
│  └────┬────┘  │  │  └────┬────┘  │  │  └────┬────┘  │
│   ┌───┴───┐   │  │   ┌───┴───┐   │  │   ┌───┴───┐   │
│   │A0│ │A1│   │  │   │A0│ │A1│   │  │   │A0│ │A1│   │
│   └──┘ └──┘   │  │   └──┘ └──┘   │  │   └──┘ └──┘   │
└───────────────┘  └───────────────┘  └───────────────┘
         ↕ channel ↕         ↕ channel ↕
```

Swarm operations:
- **Broadcast** — same kernel dispatched to all engines
- **Scatter** — data chunked across engines
- **Gather** — collect results from all engines
- **Channels** — cross-engine communication via shared mailbox

Each engine runs independently. There's no central coordinator — engines
communicate only through explicit channels. This mirrors real octopus colonies:
each animal is independent, but they can signal each other.

---

## Code Example

```flow
// Create an engine: 4 arms, 8192 floats per arm, Q4_K precision
let mut main_ids = []
let mut precisions = []
let engine = engine_create(4.0, 8192.0, "Q4_K", main_ids, precisions)

// Load data across all arms (each gets a chunk)
let data = gpu_fill(1.0, 32768)
engine_write_chunks(engine, main_ids, data, 8192.0)

// Dispatch a compute kernel to all arms
engine_dispatch_all(engine, main_ids, "stdlib/loom/kernels/math/add.spv", [0.0, 1.0, 8192.0], 32.0)
engine_exec_all(engine, main_ids)

// Gather results
let results = engine_read_all(engine, main_ids, 0.0, 8192.0)
print("Got results from {len(results)} arms")

// Check health
let health = engine_health(engine, main_ids)
print("Stable: {health.n_stable} / {health.n_main}")

// Clean up
engine_shutdown(engine, main_ids)
```

---

## Comparison

| Feature | CUDA | OpenCL | Vulkan Compute | OctoFlow Loom |
|---------|------|--------|----------------|---------------|
| GPU vendor | NVIDIA only | Multi (abandoned) | Multi (active) | Multi (Vulkan) |
| Kernel management | Manual streams | Manual queues | Manual command buffers | Automatic dispatch chains |
| Multi-GPU | Manual peer access | Manual contexts | Manual queues | Swarm: scatter/gather/channels |
| Adaptive topology | None | None | None | Parallel / Sequential / Hierarchical |
| Self-regulation | None | None | None | GPU-autonomous homeostasis |
| Multi-precision | Manual casting | Manual casting | Manual casting | Per-arm precision tiers |
| Weight management | Manual VRAM | Manual buffers | Manual memory | Three-tier: hot/warm/cold |
| Runtime adaptation | None | None | None | Brain adapt loop |
| Language | C++/PTX | C99 kernel | GLSL/HLSL | .flow (23 concepts) |
| Setup | SDK + driver + toolkit | SDK + ICD | SDK + validation layers | Single binary, zero deps |

CUDA gives you a fast GPU with manual everything. The Loom Engine gives you a
GPU that manages itself.

---

## Learn More

- [Loom Engine Reference](loom-engine.md) — full API documentation and three-tier guide
- [Loom Use Cases](loom-engine-use-cases.md) — real-world patterns across domains
- [GPU Guide](gpu-guide.md) — getting started with GPU compute in OctoFlow
