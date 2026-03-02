# distribution (L2)
stats/distribution — Probability distributions and sampling

## Functions
normal_pdf(x: float, mu: float, sigma: float) → float — Normal density
normal_cdf(x: float, mu: float, sigma: float) → float — Normal CDF
normal_inv(p: float, mu: float, sigma: float) → float — Inverse normal
uniform_random(low: float, high: float) → float — Uniform sample
normal_random(mu: float, sigma: float) → float — Normal sample
exponential_random(lambda: float) → float — Exponential sample
poisson_pmf(k: int, lambda: float) → float — Poisson PMF
binomial_pmf(k: int, n: int, p: float) → float — Binomial PMF
combinations(n: int, k: int) → float — C(n,k)
random_sample(data: array, n: int) → array — Sample without replacement
