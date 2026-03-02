# stats — Statistical Analysis

Descriptive statistics, hypothesis testing, distributions, correlations,
time series analysis, risk metrics, and extended math functions.

## Modules

| Module | Functions | Description |
|--------|-----------|-------------|
| `descriptive` | 12 | Skewness, kurtosis, percentiles, modes |
| `hypothesis` | 6 | t-tests, chi-squared, z-test |
| `distribution` | 10 | PDF/CDF, random sampling |
| `correlation` | 7 | Pearson, Spearman, covariance, regression |
| `timeseries` | 19 | SMA, EMA, RSI, MACD, Bollinger, ATR |
| `risk` | 14 | Sharpe, Sortino, drawdown, VaR |
| `math_ext` | 18 | Factorial, primes, activation functions |

## descriptive

```
use stats.descriptive
```

| Function | Description |
|----------|-------------|
| `skewness(arr)` | Statistical skewness |
| `kurtosis(arr)` | Statistical kurtosis |
| `iqr(arr)` | Interquartile range |
| `percentile(arr, p)` | p-th percentile (0-100) |
| `weighted_mean(arr, weights)` | Weighted mean |
| `trimmed_mean(arr, pct)` | Mean after trimming pct% from each end |
| `describe(arr)` | Full descriptive stats (returns map) |
| `mode(arr)` | Most frequent value |
| `geometric_mean(arr)` | Geometric mean |
| `harmonic_mean(arr)` | Harmonic mean |
| `coeff_of_variation(arr)` | Coefficient of variation (std/mean) |
| `zscore(arr)` | Z-score standardization |

```
let data = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
let sk = skewness(data)             // 0.0 (symmetric)
let ku = kurtosis(data)             // < 3 (platykurtic)
let p90 = percentile(data, 90)      // 91.0
let stats = describe(data)          // map with mean, std, min, max, etc.
```

## hypothesis

```
use stats.hypothesis
```

| Function | Description |
|----------|-------------|
| `t_test_one_sample(arr, mu0)` | One-sample t-test against hypothesized mean |
| `t_test_two_sample(arr1, arr2)` | Welch's two-sample t-test |
| `paired_t_test(arr1, arr2)` | Paired t-test for dependent samples |
| `chi_squared(observed, expected)` | Chi-squared goodness-of-fit test |
| `z_test(arr, mu0, sigma)` | Z-test (known population standard deviation) |
| `normal_cdf(x, mu, sigma)` | Normal cumulative distribution function |

```
let control = [5.1, 4.9, 5.0, 5.2, 4.8]
let treatment = [5.5, 5.3, 5.6, 5.4, 5.7]
let result = t_test_two_sample(control, treatment)
```

## distribution

```
use stats.distribution
```

| Function | Description |
|----------|-------------|
| `normal_pdf(x, mu, sigma)` | Normal probability density |
| `normal_cdf(x, mu, sigma)` | Normal cumulative distribution |
| `normal_inv(p, mu, sigma)` | Inverse normal (quantile function) |
| `uniform_random(lo, hi)` | Uniform random in [lo, hi) |
| `normal_random(mu, sigma)` | Normal random variate |
| `exponential_random(lambda)` | Exponential random variate |
| `poisson_pmf(k, lambda)` | Poisson probability mass |
| `binomial_pmf(k, n, p)` | Binomial probability mass |
| `combinations(n, k)` | Binomial coefficient (n choose k) |
| `random_sample(arr, n)` | Random sample without replacement |

## correlation

```
use stats.correlation
```

| Function | Description |
|----------|-------------|
| `pearson(x, y)` | Pearson correlation coefficient |
| `spearman(x, y)` | Spearman rank correlation |
| `rank_array(arr)` | Compute ranks of array elements |
| `covariance(x, y)` | Sample covariance |
| `linear_fit(x, y)` | Simple linear regression (slope, intercept) |
| `residuals(x, y, slope, intercept)` | Regression residuals |
| `polynomial_fit(x, y, degree)` | Polynomial curve fitting |

## timeseries

```
use stats.timeseries
```

| Function | Description |
|----------|-------------|
| `sma(arr, period)` | Simple moving average |
| `ema(arr, period)` | Exponential moving average |
| `wma(arr, period)` | Weighted moving average |
| `rsi(arr, period)` | Relative Strength Index |
| `macd(arr, fast, slow, signal)` | MACD line |
| `macd_signal(arr, fast, slow, signal_period)` | MACD signal line |
| `macd_histogram(arr, fast, slow, signal_period)` | MACD histogram |
| `bollinger(arr, period, num_std)` | Bollinger middle band |
| `bollinger_upper(arr, period, num_std)` | Bollinger upper band |
| `bollinger_lower(arr, period, num_std)` | Bollinger lower band |
| `atr(high, low, close, period)` | Average True Range |
| `vwap(price, volume)` | Volume-weighted average price |
| `returns(arr)` | Simple returns |
| `log_returns(arr)` | Log returns |
| `drawdown(arr)` | Drawdown from peak |
| `rolling(arr, period, func_name)` | Rolling window aggregation |
| `diff(arr)` | First difference |
| `cumsum(arr)` | Cumulative sum |
| `autocorrelation(arr, lag)` | Autocorrelation at given lag |

```
use stats.timeseries

let closes = csv_column(csv_read("prices.csv"), "close")
let sma_20 = sma(closes, 20)
let rsi_14 = rsi(closes, 14)
let bb_upper = bollinger_upper(closes, 20, 2.0)
```

## risk

```
use stats.risk
```

| Function | Description |
|----------|-------------|
| `sharpe_ratio(returns, risk_free_rate)` | Sharpe ratio |
| `sortino_ratio(returns, risk_free_rate)` | Sortino ratio (downside risk) |
| `max_drawdown(equity)` | Maximum drawdown |
| `calmar_ratio(returns, equity)` | Calmar ratio |
| `value_at_risk(returns, confidence)` | Value at Risk |
| `expected_shortfall(returns, confidence)` | Conditional VaR (CVaR) |
| `volatility(returns, annual)` | Annualized volatility |
| `beta(returns, benchmark)` | Beta coefficient |
| `alpha(returns, benchmark, risk_free)` | Jensen's alpha |
| `information_ratio(returns, benchmark)` | Information ratio |
| `win_rate(pnl)` | Percentage of profitable trades |
| `profit_factor(pnl)` | Gross profit / gross loss |
| `expectancy(pnl)` | Expected value per trade |
| `covariance_xy(x, y)` | Covariance |

```
use stats.risk

let equity = [10000, 10200, 10150, 10400, 10350]
let rets = returns(equity)
let sr = sharpe_ratio(rets, 0.02)
let mdd = max_drawdown(equity)
print("Sharpe: {sr}, Max DD: {mdd}")
```

## math_ext

```
use stats.math_ext
```

Constants: `PI`, `E`, `TAU`, `GOLDEN_RATIO`, `SQRT2`

| Function | Description |
|----------|-------------|
| `factorial(n)` | n! |
| `permutations(n, k)` | P(n, k) |
| `gcd(a, b)` | Greatest common divisor |
| `lcm(a, b)` | Least common multiple |
| `is_prime(n)` | Primality test |
| `fibonacci(n)` | n-th Fibonacci number |
| `power_mod(base, exp, mod)` | Modular exponentiation |
| `sigmoid(x)` | Sigmoid activation |
| `tanh_fn(x)` | Tanh activation |
| `relu(x)` | ReLU activation |
| `softplus(x)` | Softplus activation |
| `logistic(x, L, k, x0)` | Logistic function |
| `linspace(start, stop, n)` | Linearly spaced array |
| `arange(start, stop, step)` | Array with step spacing |
| `dot_product(a, b)` | Dot product |
| `magnitude(arr)` | Vector magnitude (L2 norm) |
| `normalize_vec(arr)` | Unit vector |
| `cross_product_3d(a, b)` | 3D cross product |

## Built-in Statistics

These are always available without `use`:

| Function | Description |
|----------|-------------|
| `mean(arr)` | Arithmetic mean |
| `median(arr)` | Median value |
| `stddev(arr)` | Standard deviation |
| `variance(arr)` | Variance |
| `quantile(arr, q)` | q-th quantile (0.0-1.0) |
| `correlation(a, b)` | Pearson correlation |
| `min_val(arr)` | Minimum value |
| `max_val(arr)` | Maximum value |
