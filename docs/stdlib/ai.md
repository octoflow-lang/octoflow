# ai — Artificial Intelligence

Neural networks, deep learning primitives, and AI model building blocks.

> **Status**: Foundation phase. Core GPU primitives (`gpu_matmul`, `gpu_add`,
> `gpu_scale`, `gpu_where`) already support building neural network forward
> passes. Dedicated AI modules are planned for Phase 85+.

## Available Now

Neural network building blocks using built-in GPU operations:

```
// Dense layer forward pass
let output = gpu_add(gpu_matmul(input, weights, batch, in_dim, out_dim), bias)

// ReLU activation
let zeros = gpu_fill(0.0, len(output))
let activated = gpu_where(output, output, zeros)

// Softmax (manual)
let exp_vals = gpu_exp(output)
let total = gpu_sum(exp_vals)
let probs = gpu_scale(exp_vals, 1.0 / total)
```

## Planned Modules

| Module | Description | Phase |
|--------|-------------|-------|
| `nn` | Layer definitions (dense, conv, norm) | 85 |
| `optim` | Optimizers (SGD, Adam) | 85 |
| `loss` | Loss functions (MSE, cross-entropy) | 85 |
| `train` | Training loop utilities | 86 |

## See Also

- [GPU Guide](../gpu-guide.md) — GPU operations reference
- [ml](ml.md) — Classical machine learning (KNN, regression, metrics)
