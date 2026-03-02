# OctoFlow Layer-Streaming Inference

## The Problem

Large language models don't fit in VRAM. A 70B model at Q4 quantization requires ~40GB of weight storage. A consumer GPU has 6-24GB. The standard answer is "buy a bigger GPU" or "use multiple GPUs with NVLink." Both are expensive and vendor-locked.

OctoFlow's answer: decompose the model into individual layer files, stream them through a GPU VM that holds the execution engine and state, and let the disk serve as extended model memory. The GPU processes one layer at a time. Weights are transient. Intelligence is persistent.

---

## Design Principles

**Weights are data, not state.** The model's 40GB of weights are read-only lookup tables. They don't change during inference. They don't need to be in VRAM simultaneously. Only the currently-executing layer's weights need to be present.

**Activations are state, not data.** The hidden state vector (~2-4MB) carries the model's "thinking" from layer to layer. It must persist on GPU across all 80 layers. It never leaves VRAM during a forward pass.

**The GPU VM is the execution engine.** The compiled transformer kernels, the KV cache, the activation buffers — these are the computer. The layer files on disk are the program's data. The computer is small and fixed. The data can be arbitrarily large.

**Disk is the new VRAM.** An NVMe SSD at 7 GB/s can deliver a layer's weights (~450MB) in ~65ms. GPU compute per layer takes ~100-200ms. If load time < compute time, the disk bottleneck vanishes behind double-buffered prefetching.

---

## Architecture

### One-Time: GGUF Decomposition

Convert a monolithic GGUF into per-layer files:

```
decompose_gguf("llama-70b-q4_k_m.gguf", "models/llama-70b/")

Input:  llama-70b-q4_k_m.gguf (40GB, single file)

Output: models/llama-70b/
          ├── manifest.json         (~1KB — model config, layer count, dims)
          ├── embed.bin             (~100MB — token embedding table)
          ├── lm_head.bin           (~100MB — output projection)
          ├── final_norm.bin        (~16KB — final RMSNorm weights)
          ├── layer_000.bin         (~450MB — all tensors for layer 0)
          ├── layer_001.bin         (~450MB)
          ├── ...
          └── layer_079.bin         (~450MB)
```

Each `layer_NNN.bin` contains that layer's complete tensor set packed sequentially:

```
layer_NNN.bin layout:
  ┌────────────────────────────────────┐
  │ attn_norm weights    (hidden_dim)  │  RMSNorm before attention
  │ q_proj weights       (Q4 packed)   │  Query projection
  │ k_proj weights       (Q4 packed)   │  Key projection
  │ v_proj weights       (Q4 packed)   │  Value projection
  │ o_proj weights       (Q4 packed)   │  Output projection
  │ ffn_norm weights     (hidden_dim)  │  RMSNorm before FFN
  │ gate_proj weights    (Q4 packed)   │  FFN gate
  │ up_proj weights      (Q4 packed)   │  FFN up
  │ down_proj weights    (Q4 packed)   │  FFN down
  └────────────────────────────────────┘
```

Fixed layout. No headers within the layer file. Offsets computed from manifest dimensions. Raw tensor data, ready to upload directly to GPU buffer.

### manifest.json

```json
{
  "model": "llama-70b",
  "architecture": "llama",
  "num_layers": 80,
  "hidden_dim": 8192,
  "num_heads": 64,
  "num_kv_heads": 8,
  "intermediate_dim": 28672,
  "vocab_size": 128256,
  "max_context": 8192,
  "quantization": "Q4_K_M",
  "layer_file_size": 471859200,
  "rope_theta": 500000.0,
  "rms_norm_eps": 1e-5
}
```

The manifest is the only metadata. Everything else is raw tensor bytes.

---

## VRAM Layout

```
6GB GPU (e.g., GTX 1660):

  ┌──────────────────────────────────────────────┐
  │ Persistent (loaded once, stays for entire     │
  │            generation session):               │
  │                                                │
  │  Embed weights:        ~100MB                  │
  │  LM head weights:      ~100MB                  │
  │  Final norm weights:   ~16KB                   │
  │  KV cache:             ~500MB-2GB (scales      │
  │                         with context length)   │
  │  Activation buffers:   ~20MB                   │
  │  Working buffers:      ~100MB                  │
  │  Compiled kernels:     ~10MB                   │
  ├──────────────────────────────────────────────┤
  │ Transient (double-buffered, recycled           │
  │           every layer):                        │
  │                                                │
  │  Buffer A:             ~500MB (current layer)  │
  │  Buffer B:             ~500MB (next layer)     │
  ├──────────────────────────────────────────────┤
  │ Headroom:              ~200MB (driver, OS)     │
  └──────────────────────────────────────────────┘

  Total: ~1.8GB persistent + 1GB transient + 0.2GB OS = ~3GB base
  Remaining: ~3GB available for KV cache growth
  
  At 3GB KV cache: ~2048 token context for 70B model
```

### KV Cache vs Context Length Tradeoff

The KV cache grows with context length. On a 6GB GPU running 70B, available VRAM limits context:

```
┌──────────────┬────────────┬─────────────────┐
│ KV Cache     │ Context    │ VRAM Remaining   │
├──────────────┼────────────┼─────────────────┤
│ 500MB        │ ~512 tokens│ 2.5GB free       │
│ 1GB          │ ~1024      │ 2.0GB free       │
│ 2GB          │ ~2048      │ 1.0GB free       │
│ 3GB          │ ~3072      │ 0GB (maximum)    │
└──────────────┴────────────┴─────────────────┘
```

Short context (512 tokens) is practical for chat. Long context requires more VRAM or KV cache compression (future optimization).

---

## Execution Flow

### Startup

```
fn init(model_dir):
  let manifest = load_json(model_dir + "/manifest.json")
  
  // Load persistent weights (stay in VRAM for entire session)
  let embed = gpu_upload(read_file(model_dir + "/embed.bin"))
  let lm_head = gpu_upload(read_file(model_dir + "/lm_head.bin"))
  let final_norm = gpu_upload(read_file(model_dir + "/final_norm.bin"))
  
  // Allocate transient double buffers
  let buf_a = rt_alloc(manifest.layer_file_size)
  let buf_b = rt_alloc(manifest.layer_file_size)
  
  // Allocate persistent state
  let activations = rt_alloc(manifest.hidden_dim * 4)  // float32
  let kv_cache = rt_alloc(KV_CACHE_SIZE)
  
  // Pre-compile all transformer kernels
  build_kernels()  // emits .spv files once
  
  return {manifest, embed, lm_head, final_norm,
          buf_a, buf_b, activations, kv_cache}
```

### Token Generation

```
fn generate_token(ctx, token_id):
  // Embed input token
  rt_chain_begin()
    dispatch(embed_lookup, ctx.embed, token_id, ctx.activations)
  rt_chain_end()
  
  // Stream through all layers
  stream_load(ctx.model_dir + "/layer_000.bin", ctx.buf_a)  // blocking first load
  
  for layer in 0..ctx.manifest.num_layers:
    let current = if layer % 2 == 0 then ctx.buf_a else ctx.buf_b
    let next = if layer % 2 == 0 then ctx.buf_b else ctx.buf_a
    
    // Preload next layer asynchronously
    if layer < ctx.manifest.num_layers - 1:
      async_stream_load(
        ctx.model_dir + "/layer_" + pad3(layer + 1) + ".bin",
        next
      )
    
    // GPU processes current layer
    process_layer(ctx, current, layer)
    
    // Ensure next layer load is complete before continuing
    if layer < ctx.manifest.num_layers - 1:
      await_stream_load()
  
  // Final projection and sampling
  rt_chain_begin()
    dispatch(rmsnorm, ctx.activations, ctx.final_norm, normed)
    barrier()
    dispatch(gemm, normed, ctx.lm_head, logits)
    barrier()
    dispatch(sample_greedy, logits, token_out)
  rt_chain_end()
  
  return read_token(token_out)
```

### Process Single Layer

```
fn process_layer(ctx, weight_buf, layer_idx):
  // Compute tensor offsets within the layer file
  let offsets = compute_layer_offsets(ctx.manifest)
  
  rt_chain_begin()
    // Attention block
    dispatch(rmsnorm, ctx.activations, weight_buf + offsets.attn_norm, normed)
    barrier()
    dispatch(dequant_q4, weight_buf + offsets.q_proj, q_float)
    dispatch(dequant_q4, weight_buf + offsets.k_proj, k_float)
    dispatch(dequant_q4, weight_buf + offsets.v_proj, v_float)
    barrier()
    dispatch(gemm, normed, q_float, queries)
    dispatch(gemm, normed, k_float, keys)
    dispatch(gemm, normed, v_float, values)
    barrier()
    dispatch(rope, queries, layer_idx, ctx.position)
    dispatch(rope, keys, layer_idx, ctx.position)
    barrier()
    dispatch(kv_cache_write, keys, values, ctx.kv_cache, layer_idx, ctx.position)
    barrier()
    dispatch(attention_score, queries, ctx.kv_cache, layer_idx, scores)
    barrier()
    dispatch(softmax, scores, attn_weights)
    barrier()
    dispatch(gemm, attn_weights, ctx.kv_cache_v, attn_output)
    barrier()
    dispatch(dequant_q4, weight_buf + offsets.o_proj, o_float)
    barrier()
    dispatch(gemm, attn_output, o_float, projected)
    barrier()
    dispatch(residual_add, ctx.activations, projected, ctx.activations)
    barrier()
    
    // FFN block
    dispatch(rmsnorm, ctx.activations, weight_buf + offsets.ffn_norm, normed)
    barrier()
    dispatch(dequant_q4, weight_buf + offsets.gate_proj, gate_float)
    dispatch(dequant_q4, weight_buf + offsets.up_proj, up_float)
    barrier()
    dispatch(gemm, normed, gate_float, gate_out)
    dispatch(gemm, normed, up_float, up_out)
    barrier()
    dispatch(silu_mul, gate_out, up_out, ffn_mid)
    barrier()
    dispatch(dequant_q4, weight_buf + offsets.down_proj, down_float)
    barrier()
    dispatch(gemm, ffn_mid, down_float, ffn_out)
    barrier()
    dispatch(residual_add, ctx.activations, ffn_out, ctx.activations)
  rt_chain_end()
```

---

## Double-Buffer Timing

```
Timeline for layers 0-5:

CPU:  [load L0→A]  [load L1→B]  [load L2→A]  [load L3→B]  [load L4→A]  [load L5→B]
       ~65ms        ~65ms        ~65ms        ~65ms        ~65ms        ~65ms

GPU:               [compute A]  [compute B]  [compute A]  [compute B]  [compute A]
                    Layer 0      Layer 1      Layer 2      Layer 3      Layer 4
                    ~150ms       ~150ms       ~150ms       ~150ms       ~150ms

         First layer       Overlap: load hidden behind compute
         has startup       GPU never waits (if compute > load)
         penalty

Effective: ~150ms per layer after first
Total 80 layers: 65ms + (80 × 150ms) = ~12.1 seconds per token
```

### Storage Speed Impact

```
┌──────────────┬───────────┬────────────────┬──────────────────────────────┐
│ Storage      │ Load Time │ Hidden Behind  │ Effective Per-Layer Time     │
│              │ (450MB)   │ GPU Compute?   │                              │
├──────────────┼───────────┼────────────────┼──────────────────────────────┤
│ HDD          │ ~3000ms   │ ❌ never       │ ~3000ms (240s/token)         │
│ SATA SSD     │ ~800ms    │ ❌ never       │ ~800ms (64s/token)           │
│ NVMe Gen3    │ ~130ms    │ ✅ mostly      │ ~150ms (12s/token)           │
│ NVMe Gen4    │ ~65ms     │ ✅ fully       │ ~150ms (12s/token)           │
│ NVMe Gen5    │ ~32ms     │ ✅ fully       │ ~150ms (12s/token)           │
│ RAM disk     │ ~5ms      │ ✅ fully       │ ~150ms (12s/token)           │
└──────────────┴───────────┴────────────────┴──────────────────────────────┘

Key insight: NVMe Gen3 or better → disk speed doesn't matter.
             GPU compute is the bottleneck on consumer hardware.
```

---

## Async Transfer API

The streaming requires asynchronous CPU→GPU data transfer. This is the one area that needs Rust support at the OS boundary.

### FFI Surface (minimal Rust additions)

```
// Vulkan staging buffer management
rt_staging_alloc(size)           → staging buffer handle
rt_staging_load(staging, path)   → async file read into staging buffer
rt_staging_upload(staging, gpu_buf) → async staging → device transfer
rt_staging_ready(staging)        → poll: is transfer complete?
rt_staging_wait(staging)         → block until transfer complete
rt_staging_free(staging)         → release staging buffer
```

Six FFI functions. Each takes 2-3 arguments (no arg limit issues). The staging buffer is a Vulkan host-visible buffer used as an intermediate between file I/O and device-local GPU memory.

### .flow Wrapper

```
// stdlib/llm/stream.flow

fn stream_load(path, gpu_buf):
  let staging = rt_staging_alloc(gpu_buf_size(gpu_buf))
  rt_staging_load(staging, path)
  rt_staging_wait(staging)
  rt_staging_upload(staging, gpu_buf)
  rt_staging_wait(staging)
  rt_staging_free(staging)
end

fn async_stream_start(path, gpu_buf):
  let staging = rt_staging_alloc(gpu_buf_size(gpu_buf))
  rt_staging_load(staging, path)      // non-blocking
  return staging                       // caller holds handle
end

fn async_stream_finish(staging, gpu_buf):
  rt_staging_wait(staging)             // block until file read done
  rt_staging_upload(staging, gpu_buf)  // start GPU upload
  rt_staging_wait(staging)             // block until upload done
  rt_staging_free(staging)
end
```

### Optimized: Persistent Staging Buffers

For repeated layer loading, allocate two staging buffers once and reuse:

```
fn init_stream_engine():
  let stage_a = rt_staging_alloc(LAYER_SIZE)
  let stage_b = rt_staging_alloc(LAYER_SIZE)
  return {stage_a, stage_b}
end

fn stream_layer(engine, layer_idx, gpu_buf):
  let staging = if layer_idx % 2 == 0 then engine.stage_a else engine.stage_b
  rt_staging_load(staging, layer_path(layer_idx))
  rt_staging_wait(staging)
  rt_staging_upload(staging, gpu_buf)
  rt_staging_wait(staging)
end
```

No allocation per layer. Two staging buffers recycled across all 80 layers.

---

## Decomposition Tool

```
// tools/decompose_gguf.flow
//
// Usage: octoflow decompose_gguf.flow <input.gguf> <output_dir>
//
// Converts monolithic GGUF into per-layer binary files.

fn main():
  let input_path = arg(1)
  let output_dir = arg(2)
  
  let model = gguf_parse(input_path)
  
  // Write manifest
  let manifest = {
    "model": model.name,
    "architecture": model.architecture,
    "num_layers": model.num_layers,
    "hidden_dim": model.hidden_dim,
    "num_heads": model.num_heads,
    "num_kv_heads": model.num_kv_heads,
    "intermediate_dim": model.intermediate_dim,
    "vocab_size": model.vocab_size,
    "quantization": model.quantization,
    "rope_theta": model.rope_theta,
    "rms_norm_eps": model.rms_norm_eps,
  }
  write_json(output_dir + "/manifest.json", manifest)
  
  // Extract embedding and output projection
  gguf_extract_tensor(model, "token_embd.weight", output_dir + "/embed.bin")
  gguf_extract_tensor(model, "output.weight", output_dir + "/lm_head.bin")
  gguf_extract_tensor(model, "output_norm.weight", output_dir + "/final_norm.bin")
  
  // Extract each layer
  for i in 0..model.num_layers:
    let prefix = "blk." + str(i) + "."
    let layer_tensors = [
      prefix + "attn_norm.weight",
      prefix + "attn_q.weight",
      prefix + "attn_k.weight",
      prefix + "attn_v.weight",
      prefix + "attn_output.weight",
      prefix + "ffn_norm.weight",
      prefix + "ffn_gate.weight",
      prefix + "ffn_up.weight",
      prefix + "ffn_down.weight",
    ]
    gguf_extract_tensors_packed(model, layer_tensors,
                                output_dir + "/layer_" + pad3(i) + ".bin")
    print("Layer " + str(i) + "/" + str(model.num_layers) + " extracted")
  
  print("Done. " + str(model.num_layers) + " layers decomposed to " + output_dir)
end
```

Run once per model. Output is a directory of flat binary files. No re-parsing GGUF headers during inference.

---

## Homeostasis Integration

Layer streaming creates a sustained GPU workload — exactly what Homeostasis was built for. The thermal regulation paces the dispatch loop:

```
fn generate_token_with_homeostasis(ctx, token_id):
  for layer in 0..ctx.manifest.num_layers:
    let delay = homeostasis_pace()     // returns 1ms-30ms delay
    if delay > 0:
      sleep(delay)                      // cool down between layers
    
    async_preload_next_layer(...)
    process_layer(ctx, ...)
  
  return sample(ctx)
end
```

70B inference is a 12-second sustained GPU workload per token. Over a multi-turn conversation, that's minutes of continuous load. Homeostasis prevents thermal throttling, maintaining consistent per-layer compute times across the entire generation session.

Without Homeostasis: first tokens generate in ~12s, later tokens slow to ~15-18s as GPU thermally throttles.
With Homeostasis: consistent ~12s per token for hours.

---

## Performance Projections

### GTX 1660 (6GB, 1408 cores, PCIe 3.0)

```
Per token: ~12 seconds
Tokens per minute: ~5
Practical: very slow but functional
Use case: proof of concept, short completions
```

### RTX 3060 (12GB, 3584 cores, PCIe 4.0)

```
Layers 0-19 can stay resident (~4.5GB)
Stream only layers 20-79 (60 layers)
Per token: ~4-6 seconds (faster compute + fewer streams)
Tokens per minute: ~10-15
Practical: usable for short conversations
```

### RTX 4090 (24GB, 16384 cores, PCIe 4.0)

```
Layers 0-39 stay resident (~18GB)
Stream only layers 40-79 (40 layers)
Per token: ~1-2 seconds (10x faster compute + half the streams)
Tokens per minute: ~30-60
Practical: genuinely usable for conversations
```

### 2× RTX 4090 (48GB total)

```
All 80 layers resident across both GPUs
Zero streaming, pure pipeline parallelism
Per token: ~0.5-1 second
Practical: production-grade 70B serving
```

The same OctoFlow code runs on all configurations. Only the number of resident layers changes.

---

## Partial Residency Strategy

When VRAM exceeds the minimum (~3GB) but can't hold all layers, keep the most impactful layers resident:

```
Available VRAM for layer caching: X GB
Layer file size: ~450MB

Resident layers = floor(X / 0.45)

Priority for residency:
  1. First 2-3 layers   (process every token, most accessed)
  2. Last 2-3 layers    (final refinement, most accessed)
  3. Middle layers       (stream from disk)
```

The first and last layers execute for every token and benefit most from residency. Middle layers can tolerate the streaming latency since they're accessed once per token sequentially.

```
12GB GPU, 70B model:
  Persistent: embed + lm_head + KV cache + activations = ~2GB
  Available for caching: ~9GB
  Resident layers: ~20 (first 10 + last 10)
  Streamed layers: ~60

  20 resident layers: 0ms load (already in VRAM)
  60 streamed layers: ~65ms load each (double-buffered)
  
  Effective: 20 × 150ms + 60 × 150ms = 12s (same compute, less I/O pressure)
```

---

## Future Optimizations

### KV Cache Compression

Quantize the KV cache to Q8 or Q4 during inference. Halves or quarters KV cache VRAM, allowing longer context or more resident layers.

### Layer Prefetch Prediction

For multi-token generation, the layer access pattern is completely predictable (0, 1, 2, ..., 79, 0, 1, 2, ...). Prefetch 2-3 layers ahead instead of 1. Triple-buffering.

### NVMe Direct

Bypass system RAM entirely. Use Vulkan external memory to DMA from NVMe directly into staging buffers. Eliminates one copy in the transfer pipeline.

### Hot Layer Caching

Track which layers have the highest computational impact (via gradient magnitude during training or activation variance during inference). Keep those layers resident. Stream the rest.

### Layer Fusion

Adjacent small operations (RMSNorm + dequant, residual + norm) can be fused into single kernels, reducing dispatch count per layer from ~20 to ~10-12. Fewer dispatches = faster per-layer compute = more headroom for streaming.

---

## Implementation Steps

```
Phase 1: Decomposition tool
  - GGUF parser extracts per-layer tensors
  - Writes manifest.json + flat binary files
  - Test: decompose a small model (1.5B), verify layer files match original
  
Phase 2: Streaming infrastructure
  - Implement rt_staging_* FFI functions (~50-80 lines Rust, OS boundary)
  - .flow wrappers for sync and async loading
  - Test: load a layer file, verify GPU buffer contents match

Phase 3: Double-buffered inference loop ✓ COMPLETE (v1.33)
  - decomposed_load_layer (sync) + decomposed_prefetch_layer (async) Rust builtins
  - generate_decomposed.flow: double-buffered prefetch, 245 lines
  - E2E verified: Qwen2.5-1.5B generates coherent Python code from .bin files
  - Optimized: --verbose flag, fast-path cache check, RAM-only eviction (~3x speedup)
  - Commit 39d54dc

Phase 4: Incremental model scaling
  Phase 4a: 3B model (Qwen2.5-3B Q4_K_M, ~2GB GGUF)
    - Download, decompose, verify inference
    - Validates: layer count scaling, memory management
  Phase 4b: 7B model (Qwen2.5-7B Q4_K_M, ~4.5GB GGUF)
    - Download, decompose, verify inference
    - Validates: VRAM pressure, eviction strategy effectiveness
  Phase 4c: Large model validation (70B Q4_K_M, ~40GB)
    - Full streaming: disk→GPU pipeline at scale
    - Measure: tokens/sec, VRAM usage, disk I/O, thermal behavior

Phase 5: Optimization
  - Partial residency (pin first/last layers)
  - KV cache compression
  - Homeostasis integration
  - Benchmark across GPU tiers
```

---

## The Claim

A 70B language model running on a 6GB GPU. Not through approximation, not through distillation, not through layer pruning. The full model, every parameter, every layer, producing identical output to a datacenter GPU — just slower.

The disk is the new VRAM. The GPU is the computer. OctoFlow makes it work on any Vulkan GPU, from a GTX 1660 to an RTX 4090, with the same binary and the same code. The only difference is speed.

No CUDA. No NVLink. No datacenter. The model in your SSD, the intelligence in your GPU, connected by a streaming architecture that makes model size independent of VRAM size.
