# hypothesis (L2)
stats/hypothesis — Hypothesis testing

## Functions
t_test_one_sample(data: array, mu: float) → map
  One-sample t-test against hypothesized mean
t_test_two_sample(a: array, b: array) → map
  Welch two-sample t-test
paired_t_test(a: array, b: array) → map
  Paired t-test for matched samples
chi_squared(observed: array, expected: array) → map
  Chi-squared goodness-of-fit test
z_test(data: array, mu: float, sigma: float) → map
  Z-test with known population std deviation
