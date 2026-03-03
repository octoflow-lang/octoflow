# OctoFlow LLM Support

## Overview

OctoFlow provides a GPU-native stack for LLM inference, fine-tuning, and serving. The entire pipeline — from loading model weights to serving tokens — runs as dispatch chains on any Vulkan GPU. No Python. No CUDA. No framework dependency. Vendor-independent.

This document covers four capabilities:
1. **Inference** — run GGUF models on GPU
2. **Fine-tuning** — QLoRA training as dispatch chains
3. **Serving** — multi-instance continuous token generation
4. **Architecture** — stdlib layout and kernel requirements

---

## Why OctoFlow for LLMs

Current LLM stacks have a structural problem: the GPU does the compute but the CPU orchestrates everything between operations. Python decides what to run next. PyTorch builds computation graphs. CUDA synchronizes between kernels. For every operation the GPU executes, the CPU adds overhead around it.

```
Current stack (per token):
  Python → PyTorch → CUDA kernel → sync → Python → PyTorch → CUDA kernel → sync → ...
  ~~~~~    ~~~~~~~   ~~~~~~~~~~~   ~~~~   ~~~~~    ~~~~~~~   ~~~~~~~~~~~   ~~~~
  overhead overhead  actual work   wait   overhead overhead  actual work   wait

OctoFlow (per token):
  [dispatch → barrier → dispatch → barrier → dispatch → barrier → ... → sample]
  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ one submit ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  all GPU, zero CPU involvement until token is ready
```

For small models (0.5B-7B), the overhead between operations can exceed the operation time itself. Eliminating that overhead is a direct throughput multiplier.

---

## Inference

### Loading a GGUF Model

GGUF is the standard format for quantized LLM weights (llama.cpp ecosystem). OctoFlow loads GGUF directly into GPU buffers:

```
let model = gguf_load("qwen2.5-0.5b-q4_k_m.gguf")
// Parses header, extracts tensor metadata
// Loads quantized weights directly into GPU buffers
// No intermediate CPU copy for weight data
```

The model struct contains:
- Weight buffers (quantized, read-only GPU memory)
- Model config (layer count, hidden dim, head count, vocab size)
- Tokenizer vocabulary (for encode/decode)

### Forward Pass as Dispatch Chain

A single forward pass through the transformer is one dispatch chain:

```
rt_chain_begin()
  // Token embedding
  dispatch(embed_lookup, token_ids, embed_weights, hidden_state)
  barrier()

  // Transformer layers (repeated per layer)
  for layer in 0..num_layers:
    // Attention
    dispatch(rmsnorm, hidden_state, attn_norm_weights, normed)
    barrier()
    dispatch(rope, normed, position, rotated)
    barrier()
    dispatch(dequant_q4, q_weights[layer], q_float, N)
    dispatch(dequant_q4, k_weights[layer], k_float, N)
    dispatch(dequant_q4, v_weights[layer], v_float, N)
    barrier()
    dispatch(gemm, rotated, q_float, queries)
    dispatch(gemm, rotated, k_float, keys)
    dispatch(gemm, rotated, v_float, values)
    barrier()
    dispatch(attention_score, queries, keys, kv_cache, scores)
    barrier()
    dispatch(softmax, scores, attn_weights)
    barrier()
    dispatch(gemm, attn_weights, values, attn_output)
    barrier()
    dispatch(residual_add, hidden_state, attn_output, hidden_state)
    barrier()

    // FFN
    dispatch(rmsnorm, hidden_state, ffn_norm_weights, normed)
    barrier()
    dispatch(dequant_q4, gate_weights[layer], gate_float, N)
    dispatch(dequant_q4, up_weights[layer], up_float, N)
    barrier()
    dispatch(gemm, normed, gate_float, gate_out)
    dispatch(gemm, normed, up_float, up_out)
    barrier()
    dispatch(silu_mul, gate_out, up_out, ffn_mid)
    barrier()
    dispatch(dequant_q4, down_weights[layer], down_float, N)
    barrier()
    dispatch(gemm, ffn_mid, down_float, ffn_out)
    barrier()
    dispatch(residual_add, hidden_state, ffn_out, hidden_state)
    barrier()

  // Final norm + logits
  dispatch(rmsnorm, hidden_state, final_norm, normed)
  barrier()
  dispatch(gemm, normed, lm_head, logits)
  barrier()

  // Sampling
  dispatch(softmax, logits, probs)
  barrier()
  dispatch(top_p_sample, probs, temperature, token_out)
rt_chain_end()
```

One submit. The GPU runs the entire forward pass autonomously. CPU reads the sampled token when the chain completes.

### KV Cache Management

Each inference instance maintains its own KV cache as a GPU buffer:

```
// Ring buffer: when context fills, oldest entries are overwritten
let kv_cache = gpu_alloc(num_layers × 2 × max_context × head_dim)

// Per-token: write new K/V to current position
// Attention reads all cached K/V up to current position
// Position advances after each token
// When position == max_context, wrap to 0
```

The KV cache stays on GPU permanently. No CPU reads or writes during generation. The attention kernel reads from it, the cache update kernel writes to it — all within the dispatch chain.

### Memory Requirements

| Model | Quant | Weights | KV Cache (2K ctx) | Per Instance | 6GB GPU Fits |
|---|---|---|---|---|---|
| Qwen2.5 0.5B | Q4_K_M | ~300MB | ~20MB | ~320MB | ~17 instances |
| Qwen2.5 1.5B | Q4_K_M | ~900MB | ~50MB | ~950MB | ~5 instances |
| Qwen2.5 7B | Q4_K_M | ~4.0GB | ~200MB | ~4.2GB | 1 instance |
| Llama 3.2 1B | Q4_K_M | ~600MB | ~30MB | ~630MB | ~8 instances |
| Llama 3.2 3B | Q4_K_M | ~1.8GB | ~80MB | ~1.9GB | ~2 instances |

Weights are shared across instances (read-only). Only KV cache is per-instance. Smaller models allow massively parallel inference on consumer hardware.

---

## Fine-Tuning

### QLoRA on OctoFlow

QLoRA keeps base model weights frozen in Q4 quantization and trains only small LoRA adapter matrices. This is ideal for OctoFlow because:

- Base weights stay quantized — no dequantization for storage, only for computation within the dispatch chain
- Adapter parameters are tiny (rank 16-64) — gradients, optimizer states fit easily in VRAM
- The entire forward + backward + optimizer loop is one dispatch chain

### LoRA Architecture

For each target layer (typically attention Q, K, V, and output projections):

```
Original:    output = input × W                    (W is frozen Q4)
With LoRA:   output = input × W + input × A × B   (A, B are trainable float32)

A: [hidden_dim × rank]    (e.g., 896 × 32 = ~112KB)
B: [rank × hidden_dim]    (e.g., 32 × 896 = ~112KB)
```

Total trainable parameters for Qwen2.5 0.5B at rank 32: ~10MB.
Adam optimizer states (m, v per parameter): ~20MB.
Activations cache for backward pass: ~50MB.

**Total training VRAM: ~380MB for Qwen2.5 0.5B.** Fits on any GPU.

### Training Dispatch Chain

One training batch as a single dispatch chain:

```
rt_chain_begin()
  // Forward pass (with activation caching for backward)
  for layer in 0..num_layers:
    dispatch(dequant_q4, base_weights[layer], float_weights, N)
    barrier()
    dispatch(gemm_fwd_cached, input, float_weights, output, activation_cache)
    barrier()
    dispatch(lora_fwd, input, lora_A[layer], lora_B[layer], lora_output)
    barrier()
    dispatch(add, output, lora_output, combined)
    barrier()
    // ... attention, FFN with activation caching ...

  // Loss computation
  dispatch(cross_entropy, logits, labels, loss, grad_logits)
  barrier()

  // Backward pass (reversed layers, uses cached activations)
  for layer in (num_layers-1)..0:
    // Gradients flow only through LoRA adapters
    dispatch(lora_bwd, grad_output, lora_A[layer], lora_B[layer],
             activation_cache, grad_A, grad_B)
    barrier()
    // ... attention_bwd, ffn_bwd ...

  // Gradient clipping
  dispatch(gradient_global_norm, all_grads, norm)
  barrier()
  dispatch(gradient_clip, all_grads, norm, max_norm)
  barrier()

  // Optimizer step (only LoRA parameters)
  for layer in 0..num_layers:
    dispatch(adam_step, lora_A[layer], grad_A, m_A, v_A, lr, step)
    dispatch(adam_step, lora_B[layer], grad_B, m_B, v_B, lr, step)
  barrier()
rt_chain_end()
```

One submit per training batch. Forward, backward, and optimizer all run on GPU without returning to CPU. The CPU only:
- Feeds the next batch of token IDs
- Reads the loss value (for logging)
- Checks stopping criteria

### Training Loop

```
let model = gguf_load("qwen2.5-0.5b-q4_k_m.gguf")
let lora = lora_init(model, rank: 32, target: ["q_proj", "v_proj"])
let optim = adam_init(lora.params, lr: 2e-4, weight_decay: 0.01)
let data = load_dataset("train.jsonl")

for epoch in 0..3:
  for batch in data.batches(batch_size: 4):
    let loss = train_step(model, lora, optim, batch)  // one dispatch chain
    print("loss:", loss)

  // Checkpoint
  lora_save(lora, "checkpoint_epoch_" + str(epoch) + ".bin")

// Merge adapters into base model and export
let merged = lora_merge(model, lora)
gguf_save(merged, "qwen2.5-0.5b-finetuned.gguf")
```

### Multi-Experiment Training

Same GPU VM pattern — run multiple fine-tuning experiments in parallel:

```
Shared: Qwen2.5 0.5B base weights (Q4, ~300MB, read-only)

Instance 1: LoRA rank 16 on customer support data    (~25MB state)
Instance 2: LoRA rank 32 on code generation data      (~35MB state)
Instance 3: LoRA rank 64 on medical Q&A data          (~55MB state)

Total VRAM: 300MB + 25 + 35 + 55 = ~415MB
Three fine-tuning runs simultaneously on a 6GB GPU
```

Each experiment runs its own training dispatch chain. With multi-threading, the three chains submit to separate GPU queues. Three specialized models from one training session, producing three GGUF files.

### Learning Rate Schedules

Implemented as push constants — the scheduler runs on CPU and passes the current LR as a push constant to the optimizer dispatch:

```
// Cosine schedule with warmup
fn lr_at_step(step, warmup, total, max_lr, min_lr):
  if step < warmup:
    return max_lr * (step / warmup)
  let progress = (step - warmup) / (total - warmup)
  return min_lr + 0.5 * (max_lr - min_lr) * (1.0 + cos(PI * progress))
```

No GPU kernel needed for scheduling — it's one value computed on CPU per batch.

---

## Serving

### Multi-Instance Inference Server

The GPU VM architecture enables continuous serving of multiple concurrent inference streams:

```
> octoflow serve qwen2.5-0.5b-q4_k_m.gguf --instances 30

OctoFlow LLM Server
  Model:      Qwen2.5-0.5B (Q4_K_M, 312MB)
  Instances:  30 concurrent streams
  VRAM:       912MB / 6GB (300MB weights + 30 × 20MB KV cache)
  GPU:        Any Vulkan 1.2 GPU
  Status:     serving on :8080
```

### Serving Architecture

```
┌──────────────────────────────────────────────────┐
│  GPU                                              │
│                                                    │
│  Shared weights buffer (Q4, read-only, ~300MB)    │
│                                                    │
│  ┌──────────┐ ┌──────────┐     ┌──────────┐      │
│  │Instance 1│ │Instance 2│ ... │Instance 30│      │
│  │KV cache  │ │KV cache  │     │KV cache   │      │
│  │state buf │ │state buf │     │state buf  │      │
│  └──────────┘ └──────────┘     └──────────┘      │
│                                                    │
│  [dispatch chain processes ALL instances per step] │
└──────────────────────────────────────────────────┘
         │                              ▲
         ▼ tokens out                   │ prompts in
┌──────────────────────────────────────────────────┐
│  CPU Threads                                      │
│  Thread 1: submit batch A → GPU runs              │
│  Thread 2: record batch B (pre-record next batch) │
│  Thread 3: HTTP server (receive/deliver)          │
│  Thread 4: I/O double-buffer management           │
└──────────────────────────────────────────────────┘
```

### Continuous Batching

Not all instances are at the same generation step. Some are processing long prompts, some are mid-generation, some just finished and need new tasks.

```
Batch N:
  Instances 1-20:  mid-generation (continue from KV cache)
  Instances 21-25: new prompts (prefill phase)
  Instances 26-30: finished (output ready, awaiting new prompt)

Batch N+1:
  Instances 1-20:  still generating
  Instances 21-25: now mid-generation
  Instances 26-30: received new prompts (prefill phase)
```

Indirect dispatch handles the divergence. Instances in different phases dispatch different workgroup counts. Finished instances skip compute entirely — zero waste.

### Hot-Swap Batches

Multi-threading enables zero-downtime batch transitions:

```
Thread 1: [submit batch A] [        read tokens        ] [submit batch C]
Thread 2:                  [record batch B] [submit B]   [record batch D]
GPU:      [    run A     ] [    run B    ] [   run C   ] [    run D     ]
                           ↑               ↑              ↑
                     zero gap         zero gap        zero gap
```

The GPU never idles between batches. One thread submits while the other records. Continuous token generation.

### Throughput Scaling

```
┌────────────┬────────────┬────────────────────┐
│ Instances  │ Per-stream │ Total Throughput    │
├────────────┼────────────┼────────────────────┤
│ 1          │ ~30 tok/s  │ 30 tok/s           │
│ 10         │ ~28 tok/s  │ 280 tok/s          │
│ 30         │ ~25 tok/s  │ 750 tok/s          │
│ 50         │ ~18 tok/s  │ 900 tok/s          │
└────────────┴────────────┴────────────────────┘
Sweet spot: 30-50 instances (bandwidth-bound beyond that)
```

Per-stream latency decreases slightly as instances increase (shared bandwidth). Total throughput scales near-linearly until memory bandwidth saturates.

### Agentic Serving

Each instance can run an autonomous agent loop instead of simple generation:

```
Per instance (autonomous on GPU):
  1. Read task from input buffer
  2. Forward pass → generate tokens
  3. Check: <END_ACTION> token?
     → No:  continue generating
     → Yes: write result to output buffer
            read next task, reset KV cache
            continue
```

50 agents thinking simultaneously. Each at different points in reasoning. Each receiving new tasks as they complete. All on one GPU, one dispatch chain, autonomous.

### Agentic Patterns

**Parallel tool use** — agent dispatches 10 instances with different queries, collects results simultaneously.

**Speculative execution** — 5 instances try different approaches, best result wins. Wall-clock time equals single inference.

**Multi-agent debate** — instances exchange outputs through atomic mailboxes between generation rounds.

**Swarm problem solving** — 50 instances each tackle a subproblem, coordinator instance assembles the answer.

---

## Architecture: stdlib/llm/

### Module Layout

```
stdlib/llm/
  ├── gguf.flow              — GGUF format parser and writer
  │     gguf_load()          — parse file, load weights to GPU
  │     gguf_save()          — write model to GGUF format
  │     gguf_merge_lora()    — merge LoRA adapters into base
  │
  ├── tokenizer.flow         — BPE tokenizer
  │     bpe_encode()         — text → token IDs
  │     bpe_decode()         — token IDs → text
  │
  ├── kernels/               — GPU compute kernels
  │     ├── gemm.flow        — tiled matrix multiply (shared memory)
  │     ├── softmax.flow     — fused reduce_max + exp + reduce_sum
  │     ├── rmsnorm.flow     — RMS normalization
  │     ├── rope.flow        — rotary position embedding
  │     ├── silu.flow        — SiLU activation (gate × sigmoid(gate) × up)
  │     ├── dequant.flow     — Q4_0, Q4_K_M, Q8_0 → float32
  │     ├── quant.flow       — float32 → Q4_K_M (for gguf_save)
  │     ├── embed.flow       — token embedding lookup
  │     ├── attention.flow   — multi-head attention with KV cache
  │     ├── residual.flow    — residual connection (add)
  │     └── sample.flow      — greedy, top-k, top-p sampling
  │
  ├── forward.flow           — forward pass assembly
  │     transformer_forward() — compose kernels into forward chain
  │     prefill()            — process full prompt
  │     decode_step()        — generate one token
  │
  ├── train/                 — training support
  │     ├── backward.flow    — backward pass kernels
  │     │     gemm_bwd()     — gradient w.r.t. weights and input
  │     │     attention_bwd()— attention gradient
  │     │     ffn_bwd()      — FFN gradient
  │     │     rmsnorm_bwd()  — norm gradient
  │     │     rope_bwd()     — rotary embedding gradient
  │     │
  │     ├── lora.flow        — LoRA/QLoRA support
  │     │     lora_init()    — initialize adapter matrices
  │     │     lora_fwd()     — adapter forward pass
  │     │     lora_bwd()     — adapter backward pass
  │     │     lora_save()    — checkpoint adapters
  │     │     lora_load()    — restore from checkpoint
  │     │
  │     ├── optim.flow       — optimizers
  │     │     adam_step()     — AdamW with weight decay
  │     │     sgd_step()     — SGD with momentum
  │     │     grad_clip()    — global norm clipping
  │     │
  │     ├── schedule.flow    — learning rate schedules
  │     │     cosine_lr()    — cosine annealing
  │     │     linear_lr()    — linear decay
  │     │     warmup_lr()    — linear warmup
  │     │
  │     └── data.flow        — data loading
  │           load_jsonl()   — parse training data
  │           batch_collate()— pack sequences
  │           shuffle()      — GPU-side shuffling
  │
  └── serve.flow             — inference server
        server_init()        — setup multi-instance serving
        server_step()        — process one batch across all instances
        server_loop()        — continuous serving with hot-swap
```

### Kernel Dependency Map

Build order based on dependencies:

```
Tier 0 (no dependencies — build first):
  dequant.flow       — Q4 → float32 (standalone kernel)
  embed.flow         — embedding lookup (standalone)
  rmsnorm.flow       — normalization (uses existing reduce)
  rope.flow          — rotary embedding (standalone)
  silu.flow          — activation (standalone)
  residual.flow      — addition (trivial)
  sample.flow        — sampling (uses existing argmax + prefix scan)

Tier 1 (depends on Tier 0):
  gemm.flow          — tiled matmul with shared memory
                       (THE critical kernel — everything else depends on it)

Tier 2 (depends on Tier 1):
  softmax.flow       — fused reduction (reduce_max → exp → reduce_sum)
  attention.flow     — uses gemm + softmax + KV cache
  forward.flow       — composes all Tier 0-2 into transformer forward pass

Tier 3 (depends on Tier 2 — training):
  backward.flow      — backward variants of Tier 0-2 kernels
  lora.flow          — LoRA forward/backward (uses gemm)
  optim.flow         — optimizer kernels (standalone arithmetic)

Tier 4 (depends on Tier 3 — full pipeline):
  gguf.flow          — model loading/saving
  tokenizer.flow     — BPE encode/decode
  data.flow          — training data loading
  serve.flow         — multi-instance serving
```

### Critical Path: The GEMM Kernel

GEMM (General Matrix Multiply) is the single most important kernel. Every transformer layer uses it multiple times for Q/K/V projections, attention output, and FFN up/down projections.

Current benchmark: 164 ms at 256×256 (naive implementation).
Target: <15 ms at 256×256 (tiled shared memory implementation).

The tiled GEMM design:

```
// Each workgroup computes a TILE_M × TILE_N block of the output
// Load tiles of A and B into shared memory
// Compute partial dot products
// Accumulate across K dimension

Shared memory: TILE_M × TILE_K + TILE_K × TILE_N floats
Workgroup size: TILE_M × TILE_N threads
Loop: K / TILE_K iterations per output tile

// Pseudo-dispatch:
//   For each (tile_row, tile_col) in output:
//     Load A[tile_row, 0:TILE_K] into shared memory
//     Load B[0:TILE_K, tile_col] into shared memory
//     Barrier
//     Each thread computes one element of the tile product
//     Barrier
//     Advance to next K tile
//   Write accumulated result to output buffer
```

This is the most impactful single kernel to optimize. Everything — inference speed, training speed, serving throughput — is gated by GEMM performance.

---

## Benchmarks (Current)

From audit Phase B6, measured on current hardware:

```
Dispatch overhead:    ~7.3 μs per dispatch
Element-wise ops:     0.02-0.03 ms at 100K elements
GPU fill:             0.02 ms (1K) → 4.3 ms (1M)
Reduction (sum):      0.53 ms (1K) → 4.9 ms (1M)
Prefix scan (3-pass): 0.15 ms (1K) → 3.9 ms (100K)
Matmul (naive):       0.6 ms (32²) → 19.9 ms (128²) → 164 ms (256²)
Memory throughput:    ~104 GB/s effective
```

### Estimated Inference Performance (Post-GEMM Optimization)

Projections after tiled GEMM implementation:

| Model | Single Stream | 10 Instances | 30 Instances |
|---|---|---|---|
| Qwen2.5 0.5B | ~25-35 tok/s | ~200-280 tok/s | ~500-750 tok/s |
| Qwen2.5 1.5B | ~10-15 tok/s | ~80-120 tok/s | ~200-350 tok/s |
| Llama 3.2 1B | ~15-20 tok/s | ~120-160 tok/s | ~300-500 tok/s |

These projections assume tiled GEMM achieving 10-20x over naive, on a mid-range Vulkan GPU. Actual numbers will be measured and published.

---

## Build Path

### Phase 1: Inference (minimum viable)

```
Build:
  ✅ Dispatch chain infrastructure
  ✅ Prefix scan, reduce, sort, argmax
  ✅ Atomics, shared memory, barriers
  ❌ GGUF parser
  ❌ Tiled GEMM (shared memory) ← critical path
  ❌ Dequantization kernels (Q4_0, Q4_K_M)
  ❌ RMSNorm, RoPE, SiLU, softmax, embed, residual
  ❌ Attention with KV cache
  ❌ Sampling (greedy, top-p)
  ❌ Forward pass assembly

Deliverable: single-stream inference of Qwen2.5 0.5B from GGUF
```

### Phase 2: Serving (multi-instance)

```
Build:
  ❌ Multi-instance batched dispatch
  ❌ KV cache manager (per-instance ring buffers)
  ❌ Continuous batching (instances at different phases)
  ❌ Indirect dispatch for variable instance states

Deliverable: 30+ concurrent inference streams, continuous serving
```

### Phase 3: Fine-tuning

```
Build:
  ❌ Backward pass kernels (gemm_bwd, attention_bwd, etc.)
  ❌ LoRA forward/backward
  ❌ AdamW optimizer kernel
  ❌ Gradient clipping
  ❌ Training data loader
  ❌ GGUF save (with merged LoRA)

Deliverable: QLoRA fine-tuning of 0.5B-1.5B models on consumer GPU
```

### Phase 4: Agentic

```
Build:
  ❌ Agent loop controller (per-instance termination)
  ❌ Atomic mailboxes (inter-instance communication)
  ❌ Multi-threading hot-swap
  ❌ BPE tokenizer (GPU-side)

Deliverable: 50 autonomous AI agents on single consumer GPU
```

### Kernel Reuse

Every kernel built for LLM support is infrastructure for OctoBrain / F-DHGNN:

| LLM Kernel | OctoBrain Reuse |
|---|---|
| GEMM | Hypergraph message passing |
| Softmax | Attention in graph neural network |
| RMSNorm | Layer normalization in GNN |
| Dequant/Quant | Compressed weight storage |
| LoRA backward | Plasticity gradient computation |
| Adam optimizer | Plasticity weight updates |
| KV cache | State buffer for recurrent processing |
| Attention | Graph attention mechanism |

Nothing built for LLMs is wasted. Everything compounds toward convergence.

---

## The Claim

> Run, train, and serve LLMs on any Vulkan GPU. Entire pipeline as GPU dispatch chains. No Python. No CUDA. No framework. Vendor-independent.

Consumer hardware becomes LLM infrastructure. The GTX 1660 gathering dust becomes an inference server, a fine-tuning machine, or a swarm of 50 autonomous AI agents. Same binary, any GPU.
