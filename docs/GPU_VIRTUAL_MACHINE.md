# GPU Virtual Machine

## CPU Is the Power Button

The CPU turns it on. The CPU turns it off. Everything in between — the GPU runs autonomously.

This is not a CPU virtual machine accelerated by GPU. This is a different kind of computer that happens to run programs, entirely on the GPU, with the CPU reduced to the role of a power switch.

---

## The Machine

A virtual machine needs five things: fetch instructions, decode them, execute them, manage memory, and communicate. All five can live on GPU.

**Program memory** — a GPU buffer holding instructions. The VM reads from it like a CPU reads from RAM, except the "RAM" is VRAM and the "read" is a kernel dispatch.

**Execution loop** — fetch → decode → execute → update state → repeat. Each step is a dispatch. The entire loop is a single dispatch chain. One submit, thousands of cycles, zero CPU involvement.

**Memory** — GPU buffers are the VM's heap, stack, and registers. Pre-allocate a pool at startup. The VM carves it up as needed using GPU-side allocation (prefix scan over request sizes gives memory offsets, atomics give a bump allocator).

**Branching** — the VM writes its next operation to a control buffer. The dispatch chain reads that buffer to decide which kernel runs next. The GPU makes its own branching decisions from its own data. No CPU consulted.

**Communication** — when thousands of VM instances run in parallel, they communicate through atomic buffer operations. Instance 47 writes a value. Instance 203 reads it. Producer-consumer at GPU speed, no CPU mediating.

---

## The Workarounds

Every traditional objection to "no CPU while running" has a GPU-native solution:

**"I/O needs CPU"**
The VM reads from and writes to GPU buffers. Data is pre-loaded before the chain starts. Results are read after. For continuous operation: double-buffer. The GPU writes to buffer A while the CPU (in the background) swaps buffer B. The GPU never pauses.

**"Variable execution length needs CPU"**
The VM checks a termination flag in a buffer. When done, subsequent dispatches become no-ops. The chain was recorded for maximum cycles. The VM "exits early" by zeroing its own work — the GPU keeps dispatching but each dispatch does nothing. Negligible cost.

**"Dynamic decisions need CPU"**
Indirect dispatch — the GPU reads a control buffer and sizes its own workload. The VM decides what to do next, writes the decision to a buffer, and the next dispatch reads it. Self-directing compute.

**"Memory allocation needs CPU"**
Prefix scan computes offsets from request sizes. Atomic increment gives a bump pointer. The VM manages its own memory pool from a pre-allocated GPU buffer. No syscalls, no CPU heap.

**"Inter-process communication needs CPU"**
Atomic load/store on shared buffers. Lock-free queues, mailboxes, flags — all at GPU memory bandwidth. Thousands of VM instances coordinating without any CPU involvement.

---

## The Shape of It

```
CPU:
  Power on → Load program → Submit chain → Sleep → Wake → Read results → Power off

GPU VM (runs between submit and wake):
  ┌──────────────────────────────────────────────────┐
  │  Program buffer     (instructions)                │
  │  Data buffer        (heap)                        │
  │  Stack buffer       (per-instance call stack)     │
  │  I/O buffer         (double-buffered input/output)│
  │  Control buffer     (program counter, flags)      │
  │  Allocator buffer   (atomic bump pointer)         │
  │  Mailbox buffer     (inter-VM atomic messaging)   │
  │                                                    │
  │  Fetch → Decode → Execute → Memory → Branch       │
  │    (one dispatch chain, repeating, autonomous)     │
  │                                                    │
  │  × 1,000+ instances running in parallel            │
  └──────────────────────────────────────────────────┘
```

Everything is a buffer. Every operation is a kernel. Every decision is a dispatch. The machine is self-contained.

---

## Why Parallel Changes Everything

A single VM on GPU is slower than a CPU — GPU cores have lower clock speeds and the execution is sequential per instance. That's not the point.

The point is **a thousand VMs running simultaneously.** Each GPU thread runs its own independent VM on its own data. One dispatch chain drives all of them. One submit. A thousand programs executing in parallel at GPU throughput.

A CPU runs one VM well. A GPU runs a thousand VMs at once.

This is the difference between "one smart agent" and "a thousand agents collaborating." Between "one simulation" and "a thousand scenarios explored simultaneously." Between "one neural network" and "a thousand networks adapting independently and sharing discoveries through atomic mailboxes."

---

## Multi-Threading: Zero-Cost VM Swap

CPU multi-threading doesn't add compute to the GPU. It eliminates the last bottleneck: dead time between dispatch chains.

Two CPU threads leapfrog each other:

```
Thread 1: [record chain A] [submit A] [        read results       ] [record C] [submit C]
Thread 2:                  [record B ] [submit B] [    read results] [record D]
GPU:      [              run A       ] [        run B       ] [     run C      ]
                           ↑                      ↑                  ↑
                     zero gap                zero gap           zero gap
```

Recording happens on CPU. Execution happens on GPU. Different hardware. They overlap completely. The swap is invisible to the GPU because it never stops working.

**There is no performance degradation on swap because there is nothing to swap.** Traditional VM swapping is expensive: stop → save state → load new state → resume. GPU VM swapping costs nothing:

- **Nothing to save.** The VM's state is already in GPU buffers. The results persist. The CPU reads them whenever.
- **Nothing to load.** The next chain was pre-recorded on the other thread while the GPU was running. Program, state, buffers — all already in VRAM.
- **No context switch.** The GPU has no "context" in the CPU sense. Each chain is self-contained. The GPU processes whatever is in the queue.

This means you can run **completely different programs** back to back at zero cost:

- Generation 1: 1,000 VMs running program A
- Generation 2: 1,000 VMs running program B (entirely different code)
- Generation 3: 1,000 VMs running program C (evolved from A and B's results)

Each generation is a full VM swap — different instructions, different topology, different kernels. The GPU runs at continuous maximum throughput because the chains are pre-recorded and ready.

Adding more CPU threads multiplies further:

```
Thread 1 → Queue 0: 1,000 VMs (population A)
Thread 2 → Queue 1: 1,000 VMs (population B)
Thread 3 → Queue 2: 1,000 VMs (population C)
Thread 4 → I/O (loading data, writing results, never blocking GPU)
Thread 5 → runtime compilation (evolving programs while others execute)

= 3,000+ parallel VMs across multiple Vulkan queues
+ non-blocking I/O
+ code evolution concurrent with execution
```

Modern consumer GPUs have 2-8 compute queues. Multi-threading saturates all of them.

---

## Immediate Use Case: LLM Inference at Scale

The GPU VM's first production application: hosting quantized language models for massively parallel inference.

### The Problem Today

```
Current (llama.cpp / typical):
  CPU loads weights → CPU orchestrates per layer → GPU does matmul →
  CPU manages KV cache → CPU samples token → repeat
  
  Result: 1 inference stream, ~30 tokens/sec, GPU mostly idle
```

### The GPU VM Approach

Load model weights once. Run N inference instances simultaneously, all sharing the same read-only weights.

**Example: Qwen2.5 0.5B (Q4 quantization)**

```
Shared weights:       ~300MB (loaded once, read-only)
Per-instance KV cache: ~20MB each

50 instances: 300MB + (50 × 20MB) = ~1.3GB total
Fits on any modern GPU, including integrated graphics
```

50 instances of the same model, each processing a different prompt, all driven by a single dispatch chain:

```
rt_chain_begin()
  for each transformer layer:
    dispatch(attention_kernel, weights, all_kv_caches, 50_instances)
    barrier()
    dispatch(ffn_kernel, weights, all_activations, 50_instances)
    barrier()
  dispatch(sample_kernel, all_logits, 50_instances)
rt_chain_end()
```

One submit. 50 forward passes. Shared weights, independent state per instance.

```
Single stream:  1 × 30 tokens/sec  =    30 tokens/sec
GPU VM (50):   50 × 29 tokens/sec  = 1,450 tokens/sec  (same hardware)
```

### Continuous Serving with Multi-Threading

```
Thread 1: submit batch A (50 prompts) → GPU runs inference
Thread 2: record batch B (next 50 prompts) while A runs
Thread 3: I/O — receive prompt requests, deliver completed tokens

GPU: [batch A] → [batch B] → [batch C] → ...
     zero gap, continuous serving
```

```
> octoflow serve qwen2.5-0.5b.gguf --instances 50

OctoFlow GPU VM — Inference Server
  Model:      Qwen2.5-0.5B (Q4_K_M, 312MB)
  Instances:  50 concurrent
  VRAM:       1.3GB / 8GB available
  GPU:        AMD Radeon RX 7600 (Vulkan 1.3)
  Throughput: 1,450 tokens/sec
  Status:     serving
```

No Python. No CUDA. No NVIDIA requirement. Any Vulkan GPU.

---

## Agentic Inference

Each VM instance doesn't just do single-shot inference. It runs an autonomous agent loop:

```
Per VM instance, repeating on GPU:
  1. Read task from input buffer
  2. Forward pass (all transformer layers)
  3. Sample next token → append to KV cache
  4. Check: is token <END>?
     → No:  loop to step 2 (continue generating)
     → Yes: write result to output buffer
            read next task, reset KV cache
            loop to step 1
```

50 agents, each at different points in their reasoning, all in a single dispatch chain. Instances that finish get new tasks. Instances still generating continue. Indirect dispatch handles the divergence.

### Agentic Patterns

**Parallel tool use.** Agent needs 10 database queries. Dispatch 10 instances, each with a different query prompt. Results arrive simultaneously. One logical agent, ten parallel workers.

**Speculative execution.** Run 5 instances with different prompts. Take the best result. Wall-clock time: same as one inference. Quality: best of five.

**Multi-agent debate.** Instance 1 proposes. Instance 2 critiques. Instance 3 rebuts. Instance 4 synthesizes. All in parallel, exchanging outputs through atomic mailboxes between rounds.

**Swarm problem solving.** 50 instances each tackle a different subproblem. A coordinator instance assembles the final answer. Massively parallel reasoning on a single consumer GPU.

**Evolutionary prompt refinement.** Generation 1: 50 prompt variations. Score results. Generation 2: mutations of the best prompts. Multi-threading enables zero-cost generation swaps. The system evolves its own prompts at GPU speed.

---

## Batteries Included

The GPU VM composes from capabilities that already exist in OctoFlow:

| VM Requirement | OctoFlow Capability | Status |
|---|---|---|
| Execution loop | Dispatch chains (v0.89) | ✅ Proven |
| Self-branching | Indirect dispatch (v0.90) | ✅ Proven |
| Synchronization | Timeline semaphores (v0.91) | ✅ Proven |
| Dynamic parameters | Push constants (v0.92) | ✅ Proven |
| Multi-opcode execution | Pipeline composition (v0.93) | ✅ Proven |
| Memory allocation | Prefix scan (v1.03) | ✅ BIT EXACT |
| Inter-VM communication | OpAtomicIAdd/Load/Store (v1.16) | ✅ Proven |
| Intra-VM cooperation | Shared memory + barriers (Phase 118) | ✅ Proven |
| Early termination | Indirect dispatch + control buffer | ✅ Pattern exists |
| Double-buffered I/O | Existing buffer management | ✅ Infrastructure exists |
| Matrix multiply (GEMV) | v0.97 ML inference kernels | ✅ Proven |
| Token sampling (argmax) | v1.06 argmin/argmax | ✅ BIT EXACT |
| Top-p sampling | Prefix scan + threshold | ✅ Composable |

Every row is green. The VM is an assembly problem, not an invention problem.

### Additional Kernels for LLM Inference

| Component | Description | Scope |
|---|---|---|
| GGUF parser | Read model weights + config | Binary reader in .flow |
| Dequantization | Q4_0 / Q4_K_M → float32 | Single kernel per quant type |
| GEMM | Batched matrix × matrix for attention | Largest new kernel |
| Softmax | reduce_max → sub → exp → reduce_sum → div | Composed from existing |
| RoPE | Rotary position embedding | Single kernel |
| RMSNorm | Root mean square normalization | Single kernel |
| KV cache | Per-instance ring buffer management | Buffer management in .flow |
| Agent loop | Dispatch chain with per-instance termination | Existing pattern |

---

## What Runs on the GPU VM

**LLM inference at scale** — dozens of concurrent model instances on consumer hardware, vendor-independent, continuous serving with zero-cost batch swaps.

**Massively parallel agents** — thousands of AI agents, each running their own decision loop, communicating through atomic mailboxes, one submit driving all of them.

**Parallel simulation** — a thousand Monte Carlo scenarios, game states, or scientific models, each running independently, GPU comparing results simultaneously.

**Swarm intelligence** — independent VM instances sharing discoveries through atomic buffers, emergent behavior from simple rules at GPU speed.

**Self-modifying programs** — the VM's instruction buffer is writable. Combined with runtime compilation, a program generates new GPU code while running, on GPU, without CPU.

**Population-based evolution** — multiple VM populations across multiple GPU queues via multi-threading. Winners cross-pollinate through CPU channels. Evolutionary computation at GPU scale.

---

## The Principle

The CPU is infrastructure. The GPU is the computer.

Infrastructure powers up, loads the program, and steps aside. The computer runs. When it's done, infrastructure reads the results and powers down.

Everything between power-on and power-off is GPU.

Multi-threading doesn't change this principle. It makes the infrastructure faster at feeding the computer. The power button becomes a continuous power supply. The GPU VM is what OctoFlow was always building toward — a machine where the GPU runs programs autonomously, at scale, across any vendor's hardware, with the CPU doing nothing but keeping the lights on.
