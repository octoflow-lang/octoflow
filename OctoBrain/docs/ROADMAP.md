# OctoBrain Roadmap

> What's next after Phase 29. Paused for OctoFlow language development.
> Last updated: 2026-02-28

---

## Current State (Phase 29)

| Metric | Value |
|--------|-------|
| Best perplexity | 32.62 (held-out test) |
| Parameters | 5,072 |
| Training corpus | 97K words (3 books) |
| Vocabulary | 6,890 unique → 152 prototypes |
| Architecture | 2-head attention + FFN(64) + residual + AdamW |
| Runtime | ~380s per run |
| Code size | ~3,400 lines of .flow |

---

## Phase 30-35: Near-Term (High Confidence, Proven Techniques)

These phases apply well-understood techniques that should yield measurable PPL improvements with the current architecture.

### Phase 30: Full 6-Book Corpus
**Expected PPL improvement: 5-10 points**

- Switch from `corpus_medium.txt` (97K words) to `corpus_large.txt` (244K words)
- 6 books: Alice, Looking Glass, Oz, Peter Pan, Christmas Carol, Treasure Island
- 2.5x more training data → more diverse bigrams, better generalization
- May need to increase `num_pretrain_passes` to account for larger corpus
- Prototype count will likely increase from 152 to ~200+

### Phase 31: 4 Attention Heads
**Expected PPL improvement: 2-5 points**

- Increase from 2 to 4 heads (head_dim=4 to keep total params similar)
- Or increase embed_dim to 32 with head_dim=8 (more parameters, richer features)
- More heads = more specialization patterns
- Current JS divergence ~0.19 suggests room for further specialization
- Will need separate W_Q/W_K/Adam state for heads 2 and 3

### Phase 32: Layer Normalization
**Expected: training stability improvement**

- Add LayerNorm before attention and before FFN (Pre-LN transformer style)
- `norm(x) = (x - mean) / (std + eps) * gamma + beta`
- Learnable scale (gamma) and shift (beta) per norm layer
- Stabilizes gradient magnitudes across layers
- Currently attention gradients (~0.5) much smaller than FFN gradients (~4.0)
- LayerNorm would equalize this naturally

### Phase 33: Dropout Regularization
**Expected PPL improvement: 2-5 points**

- Attention dropout: zero out random attention weights during training
- FFN dropout: zero out random hidden activations
- Embedding dropout: occasionally zero out embedding dimensions
- dropout_rate=0.1 (standard for small models)
- Should reduce overfitting gap between training and eval loss
- Need pseudo-random dropout mask (hash-based, since no `rand()` builtin)

### Phase 34: Gradient Accumulation
**Expected: effective batch size increase**

- Accumulate gradients over 4 mini-batches before each Adam step
- Effective batch size: 200 × 4 = 800 samples per step
- Smoother gradients → more stable training
- No extra memory (just accumulate into existing grad arrays)
- Can increase num_passes since each pass is cheaper (same compute, fewer Adam steps)

### Phase 35: Higher-Dimensional Embeddings
**Expected PPL improvement: 5-10 points (requires Phase 31)**

- embed_dim: 16 → 32 (or 48)
- Richer feature representation per prototype
- Parameters scale: ~20K (4x current)
- May need more training passes to converge (compensate with gradient accumulation)
- Risk: interpreter speed ceiling — larger matmuls take proportionally longer

---

## Phase 36-40: Medium-Term (Moderate Confidence)

These phases require more architectural work and may hit interpreter performance limits.

### Phase 36: Two-Layer Transformer
**Expected PPL improvement: significant (hard to quantify)**

- Stack: Attn1 → FFN1 → Attn2 → FFN2
- Residual connections between layers (already have residual infrastructure)
- Layer 1 handles local patterns, Layer 2 handles longer-range dependencies
- Parameters: ~40K (8x current)
- Risk: training time may exceed budget (~10 minutes)

### Phase 37: Positional Encoding (Learned)
**Replace fixed positional decay with learned position embeddings**

- pos_embed[position] = learnable 16-dim vector (or embed_dim)
- Add to key embeddings: K = embed @ W_K + pos_embed[relative_pos]
- Or use relative positional bias: score += pos_bias[|i-j|]
- Currently using fixed exponential decay — learned positions can adapt

### Phase 38: Vocabulary Expansion (Byte-Pair or Subword)
**Reduce OOV rate on test set**

- Current: word-level prototypes, 84.6% test vocab coverage
- BPE-like: merge frequent character pairs iteratively
- Would require re-implementing tokenization in .flow
- Significant code complexity for modest test-set gains

### Phase 39: Beam Search Generation
**Improve coherence without changing model**

- Current: greedy/sampling generation from proto distribution
- Beam search (width=3-5): track top-K hypotheses
- Should improve coherence metric significantly
- Pure generation improvement, no training changes needed

### Phase 40: Knowledge Distillation from Markov
**Use Markov as teacher, neural as student**

- Markov table gives "perfect" next-proto distribution (within its assumptions)
- Train neural network to match Markov output (not just target token)
- KL divergence loss between neural softmax and Markov distribution
- May improve learning efficiency since target is smoother than one-hot

---

## Phase 41+: Long-Term (Requires OctoFlow Improvements)

These phases likely require interpreter speedups or compilation to be practical.

### Compiled Training Loop
**100x speedup target**

- Move inner training loop to Rust while keeping .flow for model definition
- Or: JIT compile hot loops via Loom Engine compute shaders
- Would allow 200K+ gradient steps instead of ~3000
- Unlocks GPT-2 Small scale within reasonable time

### Large-Scale Corpus
**1M+ words**

- Full Project Gutenberg fiction collection
- Requires compiled training loop to be practical
- 10x more data with 100x more gradient steps could reach PPL < 20

### Attention Caching / KV Cache
**Inference speedup for generation**

- Cache key/value projections for previous positions
- Only compute Q for new position
- Standard transformer inference optimization

---

## Blocking Issues for Continued Progress

### 1. Interpreter Speed (Most Critical)
The OctoFlow interpreter processes ~1M ops/sec. Phase 29's training (30 passes × 200 samples × ~30K ops per forward+backward) takes 256 seconds. Increasing model size or training steps quickly exceeds practical time budgets.

**Solutions (in priority order):**
1. Hot-loop JIT compilation via Loom Engine GPU shaders
2. Rust-native training loop with .flow model definition
3. Interpreter optimization (bytecode compilation, reduced dispatch overhead)

### 2. Embedding Dimensionality
Current 16-dim embeddings are the bottleneck for representation quality. Increasing to 32 or 64 dims would improve features but quadratically increases computation in matmuls.

### 3. Training Data Quality
The Gutenberg texts are public domain 19th/early 20th century literature. Vocabulary and style are narrow. More diverse text (modern fiction, non-fiction, technical writing) would improve generalization.

### 4. Pseudo-Random Number Generation
Dropout and data augmentation require reliable random numbers. OctoFlow lacks a `rand()` builtin — current workaround uses hash-based pseudo-randomness which has limited quality.

---

## What NOT to Do

Based on Phase 25's analysis and subsequent discoveries:

1. **Don't increase params without better optimization.** Phase 24→28 showed that optimizer quality matters more than parameter count. Always ensure the optimizer can efficiently use the parameters before adding more.

2. **Don't add complexity before using the full corpus.** The 6-book corpus (244K words) hasn't been tested with Adam+residual yet. This is the lowest-hanging fruit.

3. **Don't try to match GPT.** The goal is to prove that the architecture works at small scale, not to reach GPT-level performance. PPL < 20 on the test set would be a strong result for 5K-20K parameters on 100K-250K words.

4. **Don't over-optimize runtime.** Phase 29 runs in ~380s which is acceptable. Only optimize if a phase exceeds 10 minutes.

---

## Dependencies on OctoFlow Language

Features that would significantly accelerate OctoBrain development:

| Feature | Impact | Status |
|---------|--------|--------|
| `rand()` builtin | Enables dropout, data augmentation | Not implemented |
| Matrix multiply builtin | 10x faster inner loops | Partial (GPU only via Loom) |
| Bytecode interpreter | 5-10x faster execution | Not started |
| In-place array ops | Reduce allocation overhead | Not implemented |
| Nested function defs | Cleaner model code | Not supported |

---

## Run Any Phase

```bash
# Phase 29 (latest, best results):
octoflow run "OctoBrain/examples/bench_residual.flow" --allow-ffi --allow-read

# Phase 28 (Adam baseline):
octoflow run "OctoBrain/examples/bench_adam.flow" --allow-ffi --allow-read

# Phase 25 (hard wall assessment, informational):
octoflow run "OctoBrain/examples/bench_alice_assessment.flow"

# Build corpus (requires Python):
cd OctoBrain/data && python build_corpus.py          # medium (3 books)
cd OctoBrain/data && python build_corpus.py --all     # large (6 books)
```
