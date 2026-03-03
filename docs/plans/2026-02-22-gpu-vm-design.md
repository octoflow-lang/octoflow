# OctoFlow GPU VM Design

**Date:** 2026-02-22
**Status:** Approved
**Mantra:** GPU is the Compute, CPU is the BIOS

## Problem

The current GPU inference pipeline manages buffers through scattered, ad-hoc caches (GPU_BUFFER_CACHE, TENSOR_CACHE, GPU_ARRAYS, GpuBuffer). Layer loading is brittle, error-prone, and tightly coupled to LLM-specific logic. There is no structured abstraction for GPU-side computation.

The GPU VM provides that structure: a general-purpose virtual machine where SSBOs are memory, dispatch chains are programs, and any workload (LLM inference, image processing, agent loops) runs as a program on the VM.

## Architecture Overview

```
CPU (BIOS)                         GPU VM (Autonomous)
-----------                        -------------------
vm_boot()          ─────────▶      Allocate SSBOs, load weights
vm_program()       ─────────▶      Build VkCommandBuffer (once)
vm_execute()       ─────────▶      vkQueueSubmit
  write embedding  ─────────▶      VM 0 register R0
  wait fence       ◀─────────      All layers + regulators complete
  read logits      ◀─────────      VM N-1 register R30
```

CPU boots the system and stays out of the way. GPU runs the entire multi-layer pipeline autonomously per submit.

## Memory Model (5 SSBO Bindings)

All memory is DEVICE_LOCAL on discrete GPUs. Persistent across dispatches within a command buffer.

### Binding 0: Register File (Per-VM)

32 registers, each 4096 floats (16 KB per register, 512 KB per VM).

| Register | Purpose |
|----------|---------|
| R0 | Primary input (hidden state) |
| R1..R29 | Scratch / intermediate values |
| R30 | Outbox (message to next VM) |
| R31 | Inbox (message from previous VM) |

Register size (4096 floats) accommodates the largest model dimension (7B: n_embd=3584) with headroom.

### Binding 1: Metrics (Per-VM)

Telemetry written by layer kernels, read by the homeostasis regulator.

```
struct LayerMetrics {
    activation_norm: f32,    // L2 norm of output hidden state
    max_abs: f32,            // max absolute activation value
    compute_ticks: f32,      // GPU timestamp counter delta
    memory_used: f32,        // bytes used by this VM's weights
    layer_idx: f32,          // self-identification
    status_flags: f32,       // 0=ok, 1=anomaly detected
}
```

### Binding 2: Heap (Shared)

Weight tensors for all layers, bump-allocated at boot. Never freed mid-program. Addressed by offset via push constants.

For Q4_K quantized weights: tensors remain quantized in the heap. Dequant happens inside compute kernels (avoids 4x memory expansion).

### Binding 3: Globals (Shared, Persistent)

Data that persists across tokens and is shared across all VMs.

- KV cache (all layers, partitioned by layer index)
- Sequence position counter
- Homeostasis running statistics (target norms, drift tracking)
- Throughput report (per-layer timing for CPU profiling)

### Binding 4: Control (Shared)

Per-VM control words written by the homeostasis regulator, read by layer kernels.

```
struct LayerControl {
    scale_factor: f32,        // activation scaling (1.0 = no change)
    eviction_priority: f32,   // 0=keep, 1=eligible for eviction
    precision_mode: f32,      // 0=FP32, 1=FP16, 2=INT8 (future)
    active: f32,              // 0=skip, 1=execute (future: layer skipping)
}
```

## Execution Model

### Boot Sequence (CPU/BIOS, runs once per model)

1. **vm_boot()** -- Allocate all 5 SSBO bindings as DEVICE_LOCAL buffers. Upload weights into Heap (binding 2). Zero-initialize all other bindings. Return VM handle.

2. **vm_program(handle, kernel_list, n_instances)** -- Build a VkCommandBuffer containing the full dispatch chain:
   - For each VM instance (0..N-1):
     - Bind instance's registers + shared heap/globals/control
     - Dispatch layer computation kernels (attention, FFN)
     - Dispatch metrics-write kernel
     - Pipeline barrier (SHADER_WRITE -> SHADER_READ)
     - Dispatch regulator kernel
     - Pipeline barrier
     - Dispatch message-copy kernel (R30 -> next VM's R31)
     - Pipeline barrier
   - Return reusable program handle

3. The command buffer is built ONCE. For subsequent tokens, CPU only writes the token embedding into VM 0's R0 and re-submits.

### Per-Token Cycle

```
CPU:  write token_embedding -> VM 0 R0
CPU:  vm_execute(program)  [single vkQueueSubmit]
GPU:  Layer 0: R31(inbox) -> attention -> FFN -> R30(outbox) + Metrics
GPU:  Regulator: read Metrics[0] -> write Control[0,1,...]
GPU:  Copy: VM0.R30 -> VM1.R31
GPU:  Layer 1: R31 -> attention -> FFN -> R30 + Metrics
GPU:  Regulator: read Metrics[1] -> adjust Control
GPU:  Copy: VM1.R30 -> VM2.R31
      ...
GPU:  Layer 27: final hidden state -> R30
CPU:  wait fence
CPU:  read VM27.R30 -> logits -> sample next token
```

### Message Format (Rich)

R30 (outbox) layout:
```
[0..n_embd-1]     hidden_state vector (the data payload)
[n_embd]          activation_norm (L2 norm of hidden state)
[n_embd+1]        max_abs_value (largest absolute activation)
[n_embd+2]        compute_ticks (GPU timestamp delta)
[n_embd+3]        layer_status (0=ok, 1=anomaly)
```

Each message carries both data and telemetry. The receiving VM uses the hidden state; the regulator uses the telemetry.

## Homeostasis Regulator

A lightweight SPIR-V kernel (single workgroup, single invocation) that runs after each layer VM completes. Three regulatory loops:

### Loop 1: Memory Pressure

Monitors total VRAM usage across all VMs. If usage exceeds 90% of budget, marks the oldest layer's weights for eviction. If under budget, enables speculative prefetch for upcoming layers.

### Loop 2: Activation Stability

Monitors hidden state norms between layers. If activations explode (norm > 2x target), writes a dampening scale_factor to the next layer's Control. If activations vanish (norm < epsilon), writes an amplifying scale_factor. Maintains running mean for drift detection.

### Loop 3: Compute Throughput

Records per-layer compute ticks in Globals for profiling. Detects anomalously slow layers. Future: adaptive precision switching for slow layers.

The regulator kernel costs ~1 microsecond per invocation. 28 invocations per token = 28 microseconds total. Negligible vs actual compute.

## .flow API

### General-Purpose (any GPU workload)

```
let vm = vm_boot()
let prog = vm_program(vm, kernels, n_instances)
vm_write_register(vm, instance_id, register_idx, data_array)
vm_execute(prog)
let result = vm_read_register(vm, instance_id, register_idx)
vm_shutdown(vm)
```

### LLM Wrapper (built on general VM)

```
let vm = llm_boot(model_path, n_layers)
let prog = llm_program(vm)
// per token:
let logits = llm_forward(prog, token_embedding)
let next_token = sample_top_p(logits, top_p, temperature)
```

## Implementation Order

### Phase 1: VM Core (Rust builtins)
- `vm_boot()` -- SSBO allocation (DEVICE_LOCAL), handle management
- `vm_write_register(vm, instance, reg, data)` -- CPU -> GPU register write
- `vm_read_register(vm, instance, reg)` -- GPU -> CPU register read
- `vm_shutdown(vm)` -- free all SSBOs

### Phase 2: Dispatch Chain
- `vm_program(vm, kernels, n_instances)` -- build command buffer
- `vm_execute(prog)` -- submit and wait
- Pipeline barrier insertion between dispatches
- Push constant binding (layer_idx, n_embd, heap offsets)

### Phase 3: Prove with Matvec
- Single VM, single kernel (matvec from register + heap weight)
- Verify: write input to R0, dispatch, read output from R1
- Compare against CPU reference

### Phase 4: Message Passing
- Multi-VM dispatch chain
- copy_register kernel: VM[N].R30 -> VM[N+1].R31
- Test: 3-VM chain, verify data flows correctly

### Phase 5: Homeostasis Regulator
- Metrics-write kernel (appended to each layer's computation)
- Regulator kernel (reads metrics, writes control)
- Test activation stability loop with synthetic exploding activations

### Phase 6: LLM Mapping
- Map transformer attention + FFN kernels into layer VMs
- Wire weights from Heap, KV from Globals
- Run Qwen2.5-1.5B through the VM (known-good model for validation)
- Scale to 3B, then 7B

### Phase 7: Evolve to Indirect Dispatch
- VkCmdDispatchIndirect for GPU-side branching
- Regulator writes dispatch parameters for adaptive execution
- Layer skipping, precision switching controlled from GPU

## What Changes in OctoFlow

| Component | Change |
|-----------|--------|
| compiler.rs | New builtins: vm_boot, vm_program, vm_execute, vm_write_register, vm_read_register, vm_shutdown |
| dispatch.rs or new vm.rs | SSBO allocation (DEVICE_LOCAL), command buffer builder, fence management |
| preflight.rs | Register new builtins |
| stdlib/gpu/vm.flow | .flow-level wrappers and LLM integration |
| stdlib/gpu/kernels/ | New kernels: copy_register, regulator, metrics_write |
| ir.flow | Emit VM-aware kernels with fixed binding layout |

## What Stays the Same

- runtime.flow -- dispatch chain infrastructure (reused internally by VM)
- gpu_run -- still works for non-VM GPU operations
- ir.flow -- SPIR-V emitter (used to build VM kernels)
- All existing .flow stdlib
- The mantra: Zero Rust, GPU first

## Success Criteria

1. vm_boot + vm_write_register + single dispatch + vm_read_register produces correct matvec result
2. 3-VM chain passes messages correctly (R30 -> R31)
3. Homeostasis regulator detects and corrects exploding activations in test
4. Qwen2.5-1.5B produces coherent output running on the GPU VM
5. 7B model runs without OOM, split-brain, or garbage output
6. CPU does zero work between vm_execute submit and fence wait
