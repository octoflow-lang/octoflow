# sampling (L2)
ai/sampling — Token sampling strategies.

## Functions
sample_greedy(logits: array) → int
  Select highest-probability token
sample_top_k(logits: array, k: int) → int
  Sample from top-k tokens
sample_top_p(logits: array, p: float) → int
  Nucleus sampling (cumulative probability p)
sample_temperature(logits: array, temp: float) → array
  Apply temperature scaling to logits
sample_min_p(logits: array, min_p: float) → int
  Sample tokens above min probability ratio
