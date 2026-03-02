# OctoFlow Standard Library Reference

The OctoFlow stdlib provides **423 modules** across **22 domains** with **945+ functions** — all written in pure `.flow`. Import any module with `use <path>`:

```flow
use array_utils
use ml/regression
use media/bmp
```

For built-in functions (available without imports), see [builtins.md](builtins.md).

---

## Table of Contents

1. [Core](#core) — Math, sorting, arrays
2. [Collections](#collections) — Stack, queue, heap, graph, set, counter
3. [String](#string) — Formatting, regex, manipulation
4. [Data](#data) — CSV, JSON, I/O, validation, transforms, pipelines
5. [Database](#database) — In-memory columnar DB
6. [Statistics](#statistics) — Descriptive, distributions, correlation, time series, hypothesis, risk
7. [Machine Learning](#machine-learning) — Regression, classification, clustering, neural nets, metrics
8. [Science](#science) — Calculus, physics, interpolation, optimization, signal processing, matrices, constants
9. [Crypto](#crypto) — Hashing, random, encoding
10. [Encoding](#encoding) — Base64, hex (byte-level)
11. [Media](#media) — BMP, GIF, AVI, MP4, WAV, H.264, TTF, image processing
12. [Terminal](#terminal) — Sixel, halfblock, kitty, digit art rendering
13. [GUI](#gui) — Window, widgets, layout, canvas, plot, themes
14. [Web](#web) — HTTP, JSON utilities, URL, server
15. [DevOps](#devops) — Config, filesystem, logging, process, templates
16. [System](#system) — Args, env, platform, memory, timer, test
17. [Time](#time) — Date/time formatting and arithmetic
18. [Game](#game) — ECS, 2D physics
19. [Algorithms](#algorithms) — Pathfinding (A*)
20. [Formats](#formats) — GGUF model file parser
21. [LLM / AI](#llm--ai) — Tokenizer, inference, sampling, weight loading
22. [Loom Engine](#loom-engine) — GPU compute kernels and dispatch

---

## Core

### `math.flow`

```
use math
```

Utility math functions beyond the builtins.

| Function | Signature | Description |
|----------|-----------|-------------|
| `min_of` | `(a, b)` | Minimum of two values |
| `max_of` | `(a, b)` | Maximum of two values |
| `clamp_val` | `(x, lo, hi)` | Clamp x to [lo, hi] |
| `lerp` | `(a, b, t)` | Linear interpolation: a + (b-a)*t |
| `map_range` | `(x, a1, a2, b1, b2)` | Map x from [a1,a2] to [b1,b2] |
| `sign` | `(x)` | -1, 0, or 1 |
| `deg_to_rad` | `(d)` | Degrees to radians |
| `rad_to_deg` | `(r)` | Radians to degrees |
| `gcd` | `(a, b)` | Greatest common divisor |
| `lcm` | `(a, b)` | Least common multiple |

### `sort.flow`

```
use sort
```

| Function | Signature | Description |
|----------|-----------|-------------|
| `insertion_sort` | `(arr)` | In-place insertion sort |
| `bubble_sort` | `(arr)` | In-place bubble sort |

### `array_utils.flow`

```
use array_utils
```

Array operations beyond the builtins. Prefixed `arr_*` names preferred.

| Function | Signature | Description |
|----------|-----------|-------------|
| `arr_contains` | `(arr, val)` | 1.0 if val in arr |
| `arr_index_of` | `(arr, val)` | Index of val, or -1.0 |
| `arr_sum` | `(arr)` | Sum of elements |
| `arr_avg` | `(arr)` | Mean of elements |
| `arr_min` | `(arr)` | Minimum element |
| `arr_max` | `(arr)` | Maximum element |
| `arr_swap` | `(arr, i, j)` | Swap elements at indices i and j |
| `arr_reverse` | `(arr)` | Reverse array in-place |
| `arr_fill` | `(arr, val)` | Fill array with val |
| `arr_search` | `(arr, target)` | Binary search (sorted array) |
| `arr_sort` | `(arr)` | Insertion sort, returns new array |
| `arr_slice` | `(arr, start, stop)` | Sub-array [start, stop) |
| `arr_unique` | `(arr)` | Remove duplicates, preserving order |
| `arr_flatten` | `(arr)` | Copy elements to new array |
| `arr_zip` | `(a, b)` | Interleave two arrays |
| `arr_count` | `(arr, val)` | Count occurrences of val |

---

## Collections

### `collections/collections.flow`

```
use collections/collections
```

Lightweight data structures using maps and arrays.

**Stack (LIFO):**

| Function | Signature | Description |
|----------|-----------|-------------|
| `stack_new` | `()` | Create stack |
| `stack_push` | `(s, val)` | Push value |
| `stack_pop` | `(s)` | Pop and return top |
| `stack_peek` | `(s)` | View top without removing |
| `stack_is_empty` | `(s)` | 1.0 if empty |

**Queue (FIFO):**

| Function | Signature | Description |
|----------|-----------|-------------|
| `queue_new` | `()` | Create queue |
| `queue_enqueue` | `(q, val)` | Add to back |
| `queue_dequeue` | `(q)` | Remove from front |
| `queue_size` | `(q)` | Number of elements |
| `queue_is_empty` | `(q)` | 1.0 if empty |

**Counter:**

| Function | Signature | Description |
|----------|-----------|-------------|
| `counter_new` | `()` | Create counter |
| `counter_add` | `(c, key)` | Increment key count |
| `counter_get` | `(c, key)` | Get count for key |
| `counter_from_array` | `(arr)` | Count all elements |

**Set:**

| Function | Signature | Description |
|----------|-----------|-------------|
| `set_new` | `()` | Create set |
| `set_add` | `(s, key)` | Add element |
| `set_has` | `(s, key)` | 1.0 if element present |
| `set_remove` | `(s, key)` | Remove element |
| `set_size` | `(s)` | Number of elements |
| `set_union` | `(s1, s2)` | Union of two sets |
| `set_intersection` | `(s1, s2)` | Intersection of two sets |
| `set_difference` | `(s1, s2)` | Elements in s1 but not s2 |

**Default Map / Pair:**

| Function | Signature | Description |
|----------|-----------|-------------|
| `dmap_new` | `(default)` | Map with default value |
| `dmap_get` | `(m, key)` | Get or return default |
| `dmap_set` | `(m, key, val)` | Set value |
| `pair` | `(a, b)` | Create pair |
| `pair_first` | `(p)` | Get first element |
| `pair_second` | `(p)` | Get second element |

### `collections/stack.flow`

```
use collections/stack
```

| Function | Signature | Description |
|----------|-----------|-------------|
| `stack_create` | `()` | Create stack (array-based) |
| `stack_push` | `(s, val)` | Push value |
| `stack_pop` | `(s)` | Pop and return top |
| `stack_peek` | `(s)` | View top |
| `stack_size` | `(s)` | Number of elements |
| `stack_is_empty` | `(s)` | 1.0 if empty |
| `stack_clear` | `(s)` | Remove all elements |

### `collections/queue.flow`

```
use collections/queue
```

O(1) enqueue and dequeue using map-based indexing.

| Function | Signature | Description |
|----------|-----------|-------------|
| `queue_create` | `()` | Create queue |
| `queue_enqueue` | `(q, val)` | Add to back |
| `queue_dequeue` | `(q)` | Remove from front |
| `queue_peek` | `(q)` | View front |
| `queue_size` | `(q)` | Number of elements |
| `queue_is_empty` | `(q)` | 1.0 if empty |
| `queue_clear` | `(q)` | Remove all elements |

### `collections/heap.flow`

```
use collections/heap
```

Min-heap (priority queue).

| Function | Signature | Description |
|----------|-----------|-------------|
| `heap_create` | `()` | Create empty min-heap |
| `heap_push` | `(h, val)` | Insert value |
| `heap_pop` | `(h)` | Remove and return minimum |
| `heap_peek` | `(h)` | View minimum |
| `heap_size` | `(h)` | Number of elements |
| `heap_is_empty` | `(h)` | 1.0 if empty |
| `heapify` | `(arr)` | Build heap from array |

### `collections/graph.flow`

```
use collections/graph
```

Weighted directed graph using maps.

| Function | Signature | Description |
|----------|-----------|-------------|
| `graph_create` | `()` | Create empty graph |
| `graph_add_node` | `(g, node)` | Add node |
| `graph_add_edge` | `(g, from, to, weight)` | Add weighted edge |
| `graph_neighbors` | `(g, node)` | Get neighbor list |
| `graph_weight` | `(g, from, to)` | Get edge weight |
| `graph_nodes` | `(g)` | List all nodes |
| `graph_bfs` | `(g, start)` | Breadth-first traversal |
| `graph_has_edge` | `(g, from, to)` | 1.0 if edge exists |

---

## String

### `string/string.flow`

```
use string/string
```

String manipulation beyond the builtins.

| Function | Signature | Description |
|----------|-----------|-------------|
| `str_center` | `(s, width, fill)` | Center-pad string |
| `str_ljust` | `(s, width, fill)` | Left-justify string |
| `str_rjust` | `(s, width, fill)` | Right-justify string |
| `str_count` | `(s, sub)` | Count substring occurrences |
| `str_reverse` | `(s)` | Reverse string |
| `str_is_digit` | `(s)` | 1.0 if all digits |
| `str_is_alpha` | `(s)` | 1.0 if all letters |
| `str_is_empty` | `(s)` | 1.0 if empty or whitespace |
| `str_lstrip` | `(s)` | Strip leading whitespace |
| `str_rstrip` | `(s)` | Strip trailing whitespace |
| `str_title` | `(s)` | Title Case |
| `str_zfill` | `(s, width)` | Zero-pad to width |

### `string/format.flow`

```
use string/format
```

| Function | Signature | Description |
|----------|-----------|-------------|
| `pad_left` | `(s, width, ch)` | Pad string on left |
| `pad_right` | `(s, width, ch)` | Pad string on right |
| `format_number` | `(n, decimals)` | Format number with decimal places |
| `format_bytes` | `(n)` | Human-readable bytes (KB, MB, GB) |
| `format_duration` | `(ms)` | Human-readable duration |

### `string/regex.flow`

```
use string/regex
```

Pattern matching and text processing. Note: the built-in `regex_match`, `regex_find`, `regex_replace`, `regex_find_all`, `regex_split` are available without import.

| Function | Signature | Description |
|----------|-----------|-------------|
| `glob_match` | `(text, pattern)` | Shell-style glob matching |
| `str_escape` | `(s, chars)` | Escape characters |
| `word_count` | `(s)` | Count words |
| `re_match` | `(text, pattern)` | Alias for regex_match |
| `re_find` | `(text, pattern)` | Alias for regex_find |
| `re_replace` | `(text, pattern, rep)` | Alias for regex_replace |
| `re_test` | `(text, pattern)` | Test if pattern matches |
| `re_split` | `(text, pattern)` | Split by pattern |

---

## Data

### `data/csv.flow`

```
use data/csv
```

RFC 4180 CSV parsing and formatting with quoted field support.

| Function | Signature | Description |
|----------|-----------|-------------|
| `csv_parse_line` | `(line)` | Parse one CSV line to field array |
| `csv_parse_text` | `(text, result_rows)` | Parse CSV text to row maps |
| `csv_format` | `(headers, rows)` | Format rows as CSV string |
| `csv_format_row` | `(headers, row_map)` | Format single row |
| `csv_read` | `(path)` | Read CSV file to row maps |
| `csv_write` | `(rows, path, headers)` | Write row maps to CSV file |
| `csv_column` | `(rows, col)` | Extract numeric column |
| `csv_column_str` | `(rows, col)` | Extract string column |
| `csv_filter` | `(rows, col, op, val)` | Filter by numeric comparison |
| `csv_filter_str` | `(rows, col, op, val)` | Filter by string comparison |
| `csv_select` | `(rows, cols)` | Select columns |
| `csv_sort` | `(rows, col, ascending)` | Sort by column |

### `data/json.flow`

```
use data/json
```

Pure `.flow` recursive-descent JSON parser. Nested objects use dot-notation keys. Arrays use indexed keys with `_len`.

| Function | Description |
|----------|-------------|
| `json_parse_flow` | Parse JSON string to map |
| `json_get_array` | Reconstruct array from indexed keys |

Note: The built-in `json_parse` and `json_stringify` are available without import.

### `data/io.flow`

```
use data/io
```

| Function | Signature | Description |
|----------|-----------|-------------|
| `load_json` | `(path)` | Read and parse JSON file |
| `save_json` | `(path, data)` | Stringify and write JSON |
| `load_text` | `(path)` | Read file as string |
| `save_text` | `(path, text)` | Write string to file |
| `load_lines` | `(path)` | Read file as line array |
| `save_lines` | `(path, lines)` | Write line array to file |
| `file_size_str` | `(path)` | Human-readable file size |

### `data/transform.flow`

```
use data/transform
```

| Function | Signature | Description |
|----------|-----------|-------------|
| `normalize` | `(arr)` | Min-max normalize to [0, 1] |
| `standardize` | `(arr)` | Z-score standardization |
| `one_hot` | `(idx, n)` | One-hot encode |
| `clip` | `(arr, lo, hi)` | Clip values to range |
| `interpolate_missing` | `(arr, sentinel)` | Fill missing values |
| `resample` | `(arr, new_len)` | Resample to new length |
| `bin_data` | `(arr, n_bins)` | Bin data into histogram |
| `scale_to_range` | `(arr, lo, hi)` | Scale to arbitrary range |

### `data/validate.flow`

```
use data/validate
```

| Function | Signature | Description |
|----------|-----------|-------------|
| `is_numeric` | `(s)` | Check if string is a valid number |
| `is_email` | `(s)` | Basic email format check |
| `is_url` | `(s)` | Basic URL format check |
| `in_range` | `(x, lo, hi)` | 1.0 if lo <= x <= hi |
| `clamp_value` | `(x, lo, hi)` | Clamp to [lo, hi] |

### `data/pipeline.flow`

```
use data/pipeline
```

| Function | Signature | Description |
|----------|-----------|-------------|
| `chain_apply` | `(arr, fn_names)` | Apply transform chain to elements |
| `batch_process` | `(arr, batch_size, fn_name)` | Process in batches |
| `parallel_merge` | `(arrays)` | Merge multiple arrays |

---

## Database

### `db/core.flow`

```
use db/core
```

In-memory columnar database using maps.

| Function | Signature | Description |
|----------|-----------|-------------|
| `db_create` | `()` | Create database |
| `db_table` | `(db, name, cols)` | Create table with columns |
| `db_tables` | `(db)` | List table names |
| `db_insert` | `(table, row)` | Insert row (map) |
| `db_count` | `(table)` | Row count |
| `db_select_column` | `(table, col)` | Get column as array |
| `db_select_row` | `(table, idx)` | Get row as map |
| `db_where` | `(table, col, op, val)` | Filter rows |
| `db_import_csv` | `(table, path)` | Import from CSV |

### `db/query.flow`

```
use db/query
```

| Function | Signature | Description |
|----------|-----------|-------------|
| `db_join` | `(t1, t2, col)` | Inner join on column |
| `db_group_by` | `(table, col, agg_col, agg_fn)` | Group and aggregate |
| `db_order_by` | `(table, col, ascending)` | Sort table by column |

### `db/schema.flow`

```
use db/schema
```

| Function | Signature | Description |
|----------|-----------|-------------|
| `db_columns` | `(table)` | Get column names |
| `db_describe` | `(table)` | Table summary (count, columns) |
| `db_rename_column` | `(table, old, new)` | Rename column |

---

## Statistics

### `stats/descriptive.flow`

```
use stats/descriptive
```

| Function | Signature | Description |
|----------|-----------|-------------|
| `skewness` | `(arr)` | Sample skewness |
| `kurtosis` | `(arr)` | Sample kurtosis |
| `iqr` | `(arr)` | Interquartile range |
| `percentile` | `(arr, p)` | p-th percentile |
| `weighted_mean` | `(vals, weights)` | Weighted average |
| `trimmed_mean` | `(arr, pct)` | Mean after trimming extremes |
| `describe` | `(arr)` | Summary stats map |
| `mode` | `(arr)` | Most frequent value |
| `geometric_mean` | `(arr)` | Geometric mean |
| `harmonic_mean` | `(arr)` | Harmonic mean |
| `coeff_of_variation` | `(arr)` | CV = stddev/mean |
| `zscore` | `(arr)` | Z-score array |

### `stats/distribution.flow`

```
use stats/distribution
```

| Function | Signature | Description |
|----------|-----------|-------------|
| `normal_pdf` | `(x, mu, sigma)` | Normal probability density |
| `normal_cdf` | `(x, mu, sigma)` | Normal cumulative distribution |
| `normal_inv` | `(p, mu, sigma)` | Inverse normal (quantile) |
| `uniform_random` | `(lo, hi)` | Uniform random float |
| `normal_random` | `(mu, sigma)` | Normal random (Box-Muller) |
| `exponential_random` | `(lambda)` | Exponential random |
| `poisson_pmf` | `(k, lambda)` | Poisson probability mass |
| `binomial_pmf` | `(k, n, p)` | Binomial probability mass |
| `combinations` | `(n, k)` | n choose k |
| `random_sample` | `(arr, n)` | Sample n elements randomly |

### `stats/correlation.flow`

```
use stats/correlation
```

| Function | Signature | Description |
|----------|-----------|-------------|
| `pearson` | `(x, y)` | Pearson correlation coefficient |
| `spearman` | `(x, y)` | Spearman rank correlation |
| `rank_array` | `(arr)` | Rank elements |
| `covariance` | `(x, y)` | Sample covariance |
| `linear_fit` | `(x, y)` | Least squares fit (slope, intercept) |
| `residuals` | `(x, y, slope, intercept)` | Fit residuals |
| `polynomial_fit` | `(x, y, degree)` | Polynomial regression |

### `stats/hypothesis.flow`

```
use stats/hypothesis
```

| Function | Signature | Description |
|----------|-----------|-------------|
| `t_test_one_sample` | `(arr, mu0)` | One-sample t-test |
| `t_test_two_sample` | `(arr1, arr2)` | Two-sample t-test |
| `paired_t_test` | `(arr1, arr2)` | Paired t-test |
| `chi_squared` | `(observed, expected)` | Chi-squared test |
| `z_test` | `(arr, mu0, sigma)` | Z-test |

### `stats/timeseries.flow`

```
use stats/timeseries
```

Trading indicators and time series analysis.

| Function | Signature | Description |
|----------|-----------|-------------|
| `sma` | `(arr, period)` | Simple moving average |
| `ema` | `(arr, period)` | Exponential moving average |
| `wma` | `(arr, period)` | Weighted moving average |
| `rsi` | `(arr, period)` | Relative strength index |
| `macd` | `(arr)` | MACD line |
| `macd_signal` | `(arr)` | MACD signal line |
| `macd_histogram` | `(arr)` | MACD histogram |
| `bollinger` | `(arr, period, k)` | Bollinger band (middle) |
| `bollinger_upper` | `(arr, period, k)` | Upper Bollinger band |
| `bollinger_lower` | `(arr, period, k)` | Lower Bollinger band |
| `atr` | `(high, low, close, period)` | Average true range |
| `vwap` | `(prices, volumes)` | Volume-weighted average price |
| `returns` | `(arr)` | Simple returns |
| `log_returns` | `(arr)` | Logarithmic returns |
| `drawdown` | `(arr)` | Drawdown from peak |
| `rolling` | `(arr, window, fn_name)` | Rolling window function |
| `diff` | `(arr)` | First differences |
| `cumsum` | `(arr)` | Cumulative sum |
| `autocorrelation` | `(arr, lag)` | Autocorrelation at lag |

### `stats/risk.flow`

```
use stats/risk
```

Financial risk metrics.

| Function | Signature | Description |
|----------|-----------|-------------|
| `covariance_xy` | `(x, y)` | Covariance |
| `sharpe_ratio` | `(returns, rf)` | Sharpe ratio |
| `sortino_ratio` | `(returns, rf)` | Sortino ratio (downside risk) |
| `max_drawdown` | `(arr)` | Maximum drawdown |
| `calmar_ratio` | `(returns, rf)` | Calmar ratio |
| `value_at_risk` | `(returns, confidence)` | Value at Risk |
| `expected_shortfall` | `(returns, confidence)` | Expected shortfall (CVaR) |
| `volatility` | `(returns)` | Annualized volatility |
| `beta` | `(asset_returns, market_returns)` | Beta coefficient |
| `alpha` | `(asset, market, rf)` | Jensen's alpha |
| `information_ratio` | `(returns, benchmark)` | Information ratio |
| `win_rate` | `(returns)` | Win rate |
| `profit_factor` | `(returns)` | Profit factor |
| `expectancy` | `(returns)` | Mathematical expectancy |

### `stats/math_ext.flow`

```
use stats/math_ext
```

Extended math and activation functions.

| Function | Signature | Description |
|----------|-----------|-------------|
| `factorial` | `(n)` | n! |
| `permutations` | `(n, r)` | P(n, r) |
| `gcd` | `(a, b)` | Greatest common divisor |
| `lcm` | `(a, b)` | Least common multiple |
| `is_prime` | `(n)` | Primality test |
| `fibonacci` | `(n)` | n-th Fibonacci number |
| `power_mod` | `(base, exp, mod)` | Modular exponentiation |
| `sigmoid` | `(x)` | Sigmoid function |
| `tanh_fn` | `(x)` | Hyperbolic tangent |
| `relu` | `(x)` | ReLU activation |
| `softplus` | `(x)` | Softplus activation |
| `logistic` | `(x, L, k, x0)` | Logistic function |
| `linspace` | `(start, stop, n)` | n evenly spaced values |
| `arange` | `(start, stop, step)` | Range with step |
| `dot_product` | `(a, b)` | Dot product |
| `magnitude` | `(v)` | Vector magnitude |
| `normalize_vec` | `(v)` | Normalize to unit vector |
| `cross_product_3d` | `(a, b)` | 3D cross product |

---

## Machine Learning

### `ml/regression.flow`

```
use ml/regression
```

| Function | Signature | Description |
|----------|-----------|-------------|
| `linear_regression` | `(x, y)` | Fit line (normal equation) |
| `ridge_regression` | `(x, y, alpha)` | L2-regularized fit |
| `predict_linear` | `(x, slope, intercept)` | Predict from linear model |
| `r_squared` | `(y_true, y_pred)` | Coefficient of determination |
| `gradient_descent_linear` | `(x, y, lr, iters)` | SGD linear regression |

### `ml/classify.flow`

```
use ml/classify
```

| Function | Signature | Description |
|----------|-----------|-------------|
| `knn_predict` | `(x_train, y_train, x_test, k)` | K-nearest neighbors |
| `logistic_regression` | `(x, y, lr, iters)` | Logistic regression fit |
| `logistic_predict` | `(x, weights)` | Predict from logistic model |
| `naive_bayes_train` | `(features, labels, n_classes)` | Naive Bayes training |
| `naive_bayes_predict` | `(x, model)` | Naive Bayes prediction |

### `ml/cluster.flow`

```
use ml/cluster
```

| Function | Signature | Description |
|----------|-----------|-------------|
| `kmeans` | `(data, k, max_iter)` | K-means clustering |
| `euclidean_distance` | `(a, b, dims)` | Euclidean distance |
| `manhattan_distance` | `(a, b, dims)` | Manhattan distance |
| `silhouette_score` | `(data, labels, k, dims)` | Cluster quality metric |

### `ml/nn.flow`

```
use ml/nn
```

Neural network building blocks.

| Function | Signature | Description |
|----------|-----------|-------------|
| `dense_forward` | `(input, weights, bias, n_in, n_out)` | Dense layer forward pass |
| `relu_forward` | `(arr)` | ReLU activation |
| `sigmoid_forward` | `(arr)` | Sigmoid activation |
| `tanh_forward` | `(arr)` | Tanh activation |
| `softmax` | `(arr)` | Softmax function |
| `cross_entropy_loss` | `(pred, target)` | Cross-entropy loss |
| `mse_loss` | `(pred, target)` | Mean squared error loss |
| `init_weights` | `(n_in, n_out)` | Xavier weight initialization |
| `init_bias` | `(n)` | Zero bias initialization |
| `sgd_update` | `(params, grads, lr)` | SGD parameter update |
| `dropout` | `(arr, rate)` | Dropout regularization |
| `batch_norm` | `(arr, gamma, beta)` | Batch normalization |

### `ml/tree.flow`

```
use ml/tree
```

| Function | Signature | Description |
|----------|-----------|-------------|
| `gini_impurity` | `(labels, n_classes)` | Gini impurity |
| `decision_stump` | `(x, y, threshold)` | Single-split decision |
| `find_best_split` | `(x, y, n_classes)` | Best split threshold |

### `ml/ensemble.flow`

```
use ml/ensemble
```

| Function | Signature | Description |
|----------|-----------|-------------|
| `bagging_predict` | `(predictions)` | Majority vote ensemble |
| `weighted_vote` | `(predictions, weights)` | Weighted majority vote |
| `bootstrap_sample` | `(data, size)` | Random bootstrap sample |
| `bagging_predict_array` | `(pred_arrays, n_samples)` | Per-sample bagging |
| `bootstrap_indices` | `(n, size)` | Random index sample |

### `ml/metrics.flow`

```
use ml/metrics
```

| Function | Signature | Description |
|----------|-----------|-------------|
| `accuracy` | `(y_true, y_pred)` | Classification accuracy |
| `precision` | `(y_true, y_pred)` | Precision (binary) |
| `recall` | `(y_true, y_pred)` | Recall (binary) |
| `f1_score` | `(y_true, y_pred)` | F1 score (binary) |
| `mse` | `(y_true, y_pred)` | Mean squared error |
| `rmse` | `(y_true, y_pred)` | Root mean squared error |
| `mae` | `(y_true, y_pred)` | Mean absolute error |
| `confusion_matrix` | `(y_true, y_pred, n)` | Confusion matrix |
| `mean_absolute_percentage_error` | `(y_true, y_pred)` | MAPE |

### `ml/preprocess.flow`

```
use ml/preprocess
```

| Function | Signature | Description |
|----------|-----------|-------------|
| `train_test_split` | `(data, ratio, train_out, test_out)` | Split data |
| `shuffle_array` | `(arr)` | Random shuffle |
| `minmax_scale` | `(arr)` | Scale to [0, 1] |
| `zscore_scale` | `(arr)` | Z-score normalization |
| `feature_scale` | `(arr, method)` | Scale by method name |
| `encode_labels` | `(labels)` | Encode string labels to ints |
| `impute_missing` | `(arr, sentinel, method)` | Fill missing values |

### `ml/linalg.flow`

```
use ml/linalg
```

Matrix operations (row-major flat arrays).

| Function | Signature | Description |
|----------|-----------|-------------|
| `mat_create` | `(rows, cols, val)` | Create matrix filled with val |
| `mat_identity` | `(n)` | Identity matrix |
| `mat_get` | `(m, cols, r, c)` | Get element |
| `mat_set` | `(m, cols, r, c, val)` | Set element |
| `mat_mul` | `(a, b, m, n, p)` | Matrix multiplication |
| `mat_transpose` | `(m, rows, cols)` | Transpose |
| `mat_add` | `(a, b)` | Element-wise add |
| `mat_scale` | `(m, s)` | Scalar multiply |
| `mat_det_2x2` | `(m)` | 2x2 determinant |
| `mat_det_3x3` | `(m)` | 3x3 determinant |
| `mat_inverse_2x2` | `(m)` | 2x2 inverse |
| `outer_product` | `(a, b)` | Outer product |
| `mat_trace` | `(m, n)` | Matrix trace |
| `solve_2x2` | `(a, b)` | Solve 2x2 linear system |

---

## Science

### `science/calculus.flow`

```
use science/calculus
```

| Function | Signature | Description |
|----------|-----------|-------------|
| `derivative` | `(arr, dx)` | Numerical derivative (central diff) |
| `integrate_trapz` | `(arr, dx)` | Trapezoidal integration |
| `cumulative_trapz` | `(arr, dx)` | Cumulative trapezoidal integral |
| `second_derivative` | `(arr, dx)` | Second derivative |

### `science/physics.flow`

```
use science/physics
```

| Function | Signature | Description |
|----------|-----------|-------------|
| `integrate_euler` | `(state, dt, accel_fn_name)` | Euler integration |
| `integrate_rk4` | `(state, dt, accel_fn_name)` | 4th-order Runge-Kutta |
| `spring_damper` | `(x, v, k, c, m)` | Spring-damper system |
| `projectile` | `(v0, angle, g, dt)` | Projectile trajectory |
| `kinetic_energy` | `(m, v)` | KE = 0.5*m*v^2 |
| `potential_energy` | `(m, g, h)` | PE = m*g*h |
| `gravitational_force` | `(m1, m2, r)` | Newton's gravity |
| `wave_equation_1d` | `(u, c, dx, dt, steps)` | 1D wave equation |

### `science/interpolate.flow`

```
use science/interpolate
```

| Function | Signature | Description |
|----------|-----------|-------------|
| `linear_interp` | `(x_data, y_data, x)` | Linear interpolation at point |
| `linear_interp_array` | `(x_data, y_data, x_query)` | Interpolate array of points |
| `bilinear_interp` | `(z, nx, ny, x, y)` | Bilinear interpolation on grid |
| `nearest_interp` | `(x_data, y_data, x)` | Nearest-neighbor interpolation |

### `science/optimize.flow`

```
use science/optimize
```

| Function | Signature | Description |
|----------|-----------|-------------|
| `gradient_descent` | `(x0, lr, max_iter, grad_fn_name)` | 1D gradient descent |
| `golden_section` | `(a, b, tol, fn_name)` | Golden section search |
| `newton_raphson` | `(x0, tol, max_iter, f_name, fp_name)` | Root finding |
| `bisection` | `(a, b, tol, max_iter, fn_name)` | Bisection root finding |
| `integrate_trapezoid` | `(a, b, n, fn_name)` | Numerical integration |
| `integrate_simpson` | `(a, b, n, fn_name)` | Simpson's rule integration |
| `differentiate` | `(x, h, fn_name)` | Numerical differentiation |

### `science/signal.flow`

```
use science/signal
```

| Function | Signature | Description |
|----------|-----------|-------------|
| `convolve` | `(signal, kernel)` | 1D convolution |
| `moving_avg_filter` | `(signal, window)` | Moving average filter |
| `gaussian_kernel` | `(size, sigma)` | Gaussian kernel |
| `hamming_window` | `(n)` | Hamming window |
| `hanning_window` | `(n)` | Hanning window |
| `blackman_window` | `(n)` | Blackman window |
| `cross_correlate` | `(a, b)` | Cross-correlation |
| `bandpass` | `(signal, lo, hi, sr)` | Bandpass filter |
| `envelope` | `(signal)` | Signal envelope |
| `zero_crossings` | `(signal)` | Count zero crossings |
| `peak_detect` | `(signal, threshold)` | Detect peaks above threshold |

### `science/matrix.flow`

```
use science/matrix
```

Extended matrix operations.

| Function | Signature | Description |
|----------|-----------|-------------|
| `mat_norm` | `(a, n)` | Frobenius norm |
| `mat_diag` | `(a, n)` | Extract diagonal |
| `mat_from_diag` | `(d)` | Create diagonal matrix |
| `mat_solve_triangular` | `(L, b, n)` | Solve lower-triangular system |
| `mat_solve_upper` | `(U, b, n)` | Solve upper-triangular system |
| `mat_row_norm` | `(a, n, row)` | Row L2 norm |
| `mat_col_norm` | `(a, n, col)` | Column L2 norm |

### `science/constants.flow`

```
use science/constants
```

Mathematical and physical constants (no functions, just `let` bindings):

| Constant | Value |
|----------|-------|
| `PI` | 3.14159265358979 |
| `E` | 2.71828182845905 |
| `TAU` | 6.28318530717959 |
| `GOLDEN_RATIO` | 1.61803398874989 |
| `SQRT2` | 1.41421356237310 |
| `LN2`, `LN10` | Natural log of 2, 10 |
| `SPEED_OF_LIGHT` | 299,792,458 m/s |
| `PLANCK` | 6.626e-34 J*s |
| `BOLTZMANN` | 1.381e-23 J/K |
| `AVOGADRO` | 6.022e23 /mol |
| `GRAVITY` | 9.80665 m/s^2 |
| `GAS_CONSTANT` | 8.31446 J/(mol*K) |

---

## Crypto

### `crypto/hash.flow`

```
use crypto/hash
```

Non-cryptographic hash functions.

| Function | Signature | Description |
|----------|-----------|-------------|
| `djb2` | `(s)` | DJB2 string hash |
| `fnv1a` | `(s)` | FNV-1a string hash |
| `checksum` | `(arr)` | Simple additive checksum |
| `crc_simple` | `(arr)` | Simple CRC |

Note: Built-in `sha256(s)` is available without import.

### `crypto/random.flow`

```
use crypto/random
```

| Function | Signature | Description |
|----------|-----------|-------------|
| `random_float` | `(lo, hi)` | Random float in [lo, hi) |
| `random_int` | `(lo, hi)` | Random integer in [lo, hi] |
| `random_hex` | `(n)` | Random hex string of n chars |
| `uuid_v4` | `()` | UUID v4 string |
| `random_token` | `(n)` | Random alphanumeric token |
| `random_choice` | `(arr)` | Random element from array |
| `random_shuffle` | `(arr)` | Shuffle array in-place |

### `crypto/encoding.flow`

```
use crypto/encoding
```

Aliases for built-in encoding functions: `base64_encode`, `base64_decode`, `hex_encode`, `hex_decode`. Prefer using the builtins directly.

---

## Encoding

### `encoding/base64.flow`

```
use encoding/base64
```

Byte-level Base64 encoding/decoding (byte arrays, not strings).

| Function | Signature | Description |
|----------|-----------|-------------|
| `b64_char` | `(idx)` | Index to Base64 character |
| `b64_val` | `(ch)` | Base64 character to index |
| `b64_encode` | `(bytes)` | Encode byte array to Base64 string |
| `b64_decode` | `(s)` | Decode Base64 string to byte array |

### `encoding/hex.flow`

```
use encoding/hex
```

Byte-level hex encoding/decoding.

| Function | Signature | Description |
|----------|-----------|-------------|
| `hex_nibble` | `(val)` | Nibble (0-15) to hex char |
| `hex_val` | `(ch)` | Hex char to value |
| `bytes_to_hex` | `(bytes)` | Byte array to hex string |
| `hex_to_bytes` | `(s)` | Hex string to byte array |

---

## Media

### `media/bmp.flow`

```
use media/bmp
```

BMP image decoder. Supports 8-bit palettized, 24-bit BGR, 32-bit BGRA.

| Function | Signature | Description |
|----------|-----------|-------------|
| `bmp_parse` | `(bytes)` | Parse BMP header only |
| `bmp_decode` | `(bytes)` | Full decode to pixel arrays |

Output: `bmp_r`, `bmp_g`, `bmp_b`, `bmp_a` module-level arrays (top-down, left-to-right, 0-255 floats).

### `media/gif.flow`

```
use media/gif
```

GIF87a/GIF89a decoder with LZW decompression.

| Function | Signature | Description |
|----------|-----------|-------------|
| `gif_decode` | `(bytes)` | Decode GIF to pixel/frame data |
| `gif_extract_r` | `(pixels, frame, w, h)` | Extract red channel for frame |
| `gif_extract_g` | `(pixels, frame, w, h)` | Extract green channel for frame |
| `gif_extract_b` | `(pixels, frame, w, h)` | Extract blue channel for frame |

### `media/gif_encode.flow`

```
use media/gif_encode
```

GIF89a encoder with LZW compression and animation support.

| Function | Signature | Description |
|----------|-----------|-------------|
| `gif_encode` | `(pixels, w, h, path)` | Encode single-frame GIF |
| `gif_encode_frames` | `(pixels, delays, n, w, h, path)` | Encode animated GIF |

### `media/avi.flow`

```
use media/avi
```

AVI/RIFF container parser. Extracts video frame locations.

| Function | Signature | Description |
|----------|-----------|-------------|
| `avi_parse` | `(bytes)` | Parse AVI metadata and frame index |
| `avi_get_frame` | `(bytes, idx)` | Extract frame bytes |

Output: `avi_offsets` module-level array (flat offset/length pairs).

### `media/mp4.flow`

```
use media/mp4
```

MP4/ISO BMFF container demuxer. Extracts H.264 sample locations.

| Function | Signature | Description |
|----------|-----------|-------------|
| `mp4_parse` | `(bytes)` | Parse MP4 and build sample table |
| `mp4_get_sample` | `(bytes, idx)` | Extract sample bytes |
| `mp4_find_box` | `(bytes, start, end, tag)` | Find ISOBMFF box by FourCC |

Output: `mp4_offsets`, `mp4_sps`, `mp4_pps` module-level arrays.

### `media/h264.flow`

```
use media/h264
```

H.264 Baseline I-frame decoder. Use with `mp4.flow` for input.

| Function | Signature | Description |
|----------|-----------|-------------|
| `h264_decode_sps` | `(sps_bytes)` | Decode Sequence Parameter Set |
| `h264_decode_pps` | `(pps_bytes)` | Decode Picture Parameter Set |
| `h264_decode_idr` | `(sample, nal_len_size)` | Decode IDR I-frame to RGB |
| `h264_yuv_to_rgb` | `(y, cb, cr, w, h, rgb)` | YCbCr to RGB conversion |
| `h264_idct4x4` | `(coeffs, offset, out)` | 4x4 inverse DCT |

Output: `h264_dims` module-level array after SPS decode.

### `media/wav.flow`

```
use media/wav
```

WAV audio parser. Supports 8-bit and 16-bit PCM.

| Function | Signature | Description |
|----------|-----------|-------------|
| `wav_parse` | `(bytes)` | Parse WAV header metadata |
| `wav_get_samples` | `(bytes, info, max)` | Extract PCM samples (normalized -1..1) |

Output: `wav_samples` module-level array.

### `media/ttf.flow`

```
use media/ttf
```

TrueType font parser. Extracts glyph bezier contours for rendering.

| Function | Signature | Description |
|----------|-----------|-------------|
| `ttf_parse_all_ascii` | `(path)` | Parse TTF and extract ASCII glyphs |

Output: `ttf_seg_data` (bezier segments), `ttf_seg_counts`, `ttf_seg_offsets`, `ttf_metrics`, `ttf_font_info`.

### `media/image_util.flow`

```
use media/image_util
```

Image manipulation on flat channel arrays.

| Function | Signature | Description |
|----------|-----------|-------------|
| `img_get_pixel` | `(r, g, b, w, x, y, result)` | Get pixel RGB |
| `img_set_pixel` | `(r, g, b, w, x, y, pr, pg, pb)` | Set pixel RGB |
| `img_fill` | `(r, g, b, w, h, pr, pg, pb)` | Fill with solid color |
| `img_copy` | `(src_r, src_g, src_b, dst_r, dst_g, dst_b, count)` | Copy channels |
| `img_crop` | `(r, g, b, w, cx, cy, cw, ch, out_r, out_g, out_b)` | Crop region |
| `img_paste` | `(dst_r, ..., src_r, ..., src_w, src_h, px, py)` | Paste image |
| `img_flip_h` | `(r, g, b, w, h)` | Flip horizontal |
| `img_flip_v` | `(r, g, b, w, h)` | Flip vertical |
| `img_grayscale` | `(r, g, b, count)` | Convert to grayscale |
| `img_brightness` | `(r, g, b, count, amount)` | Adjust brightness |
| `img_contrast` | `(r, g, b, count, factor)` | Adjust contrast |
| `img_mse` | `(r1, g1, b1, r2, g2, b2, count)` | Mean squared error |
| `img_psnr` | `(mse_val)` | Peak signal-to-noise ratio |

### `media/image_filter.flow`

```
use media/image_filter
```

Convolution filters for image processing.

| Function | Signature | Description |
|----------|-----------|-------------|
| `img_convolve` | `(r, g, b, w, h, kernel, ksize, out_r, out_g, out_b)` | Generic convolution |
| `img_blur_box` | `(r, g, b, w, h, out_r, out_g, out_b)` | 3x3 box blur |
| `img_sharpen` | `(r, g, b, w, h, out_r, out_g, out_b)` | 3x3 sharpen |
| `img_edge_detect` | `(r, g, b, w, h, out_r, out_g, out_b)` | Laplacian edge detect |
| `img_threshold` | `(r, g, b, count, thresh)` | Binary threshold |
| `img_invert` | `(r, g, b, count)` | Invert colors |

---

## Terminal

Terminal image renderers for different protocols.

### `terminal/sixel.flow`

```
use terminal/sixel
```

| Function | Description |
|----------|-------------|
| `sixel_render(w, h, r, g, b)` | Render image as Sixel graphics |

### `terminal/halfblock.flow`

```
use terminal/halfblock
```

| Function | Description |
|----------|-------------|
| `halfblock_render(w, h, r, g, b)` | Render with ANSI truecolor halfblocks |

### `terminal/kitty.flow`

```
use terminal/kitty
```

| Function | Description |
|----------|-------------|
| `kitty_render(w, h, r, g, b)` | Render via Kitty graphics protocol (GPU-accelerated) |

### `terminal/digits.flow`

```
use terminal/digits
```

| Function | Description |
|----------|-------------|
| `digits_render(w, h, r, g, b)` | Render as ASCII digit art |

### `terminal/render.flow`

```
use terminal/render
```

Unified renderer that dispatches to the best available protocol.

| Function | Description |
|----------|-------------|
| `term_render(w, h, r, g, b, mode)` | Render with specified mode |
| `term_up(n)` | Move cursor up n lines |

---

## GUI

### `gui/widgets.flow`

```
use gui/widgets
```

Complete widget toolkit with 15+ widget types.

**Window:**

| Function | Description |
|----------|-------------|
| `gui_init(w, h, title)` | Create window |
| `gui_update()` | Process events and render |
| `gui_running()` | 1.0 while window is open |

**Widgets:**

| Function | Description |
|----------|-------------|
| `gui_panel(x, y, w, h)` | Panel container |
| `gui_label(x, y, text)` | Text label |
| `gui_button(x, y, w, h, text)` | Button |
| `gui_checkbox(x, y, text)` | Checkbox |
| `gui_slider(x, y, w, min, max, init)` | Slider |
| `gui_textinput(x, y, w)` | Text input field |
| `gui_progress(x, y, w)` | Progress bar |
| `gui_radio(x, y, text, group)` | Radio button |
| `gui_separator(x, y, w)` | Horizontal separator |
| `gui_listbox(x, y, w, h)` | List box |
| `gui_spinbox(x, y, w, min, max, init, step)` | Spin box |
| `gui_dropdown(x, y, w)` | Dropdown menu |
| `gui_tabs(x, y, w, h)` | Tab container |
| `gui_treeview(x, y, w, h)` | Tree view |
| `gui_tooltip(target_id, text)` | Tooltip |

**State:**

| Function | Description |
|----------|-------------|
| `gui_clicked(id)` | 1.0 if button clicked |
| `gui_checked(id)` | 1.0 if checkbox checked |
| `gui_slider_value(id)` | Current slider value |
| `gui_get_text(id)` | Get text input value |
| `gui_set_text(id, text)` | Set text content |
| `gui_set_checked(id, val)` | Set checkbox state |
| `gui_set_slider(id, val)` | Set slider position |
| `gui_set_visible(id, val)` | Show/hide widget |
| `gui_set_enabled(id, val)` | Enable/disable widget |
| `gui_set_progress(id, val)` | Set progress bar value |

### `gui/layout.flow`

```
use gui/layout
```

Automatic layout managers: `vstack`, `hstack`, `grid`.

```flow
let lay = vstack(10, 10, 5)
vstack_label(lay, "Name:")
let inp = vstack_textinput(lay, 200)
let btn = vstack_button(lay, 100, 30, "Submit")
```

### `gui/canvas.flow`

```
use gui/canvas
```

Drawing surface for custom graphics.

| Function | Description |
|----------|-------------|
| `gui_canvas(x, y, w, h)` | Create canvas widget |
| `gui_canvas_clear(id)` | Clear canvas |
| `gui_canvas_line(id, x1, y1, x2, y2, r, g, b)` | Draw line |
| `gui_canvas_rect(id, x, y, w, h, r, g, b)` | Draw rectangle |
| `gui_canvas_fill(id, x, y, w, h, r, g, b)` | Filled rectangle |
| `gui_canvas_circle(id, cx, cy, r, cr, cg, cb)` | Draw circle |
| `gui_canvas_fill_circle(id, cx, cy, r, cr, cg, cb)` | Filled circle |
| `gui_canvas_pixel(id, x, y, r, g, b)` | Set single pixel |

### `gui/plot.flow`

```
use gui/plot
```

Data visualization on canvas widgets.

| Function | Description |
|----------|-------------|
| `plot_create(canvas_id, x_min, x_max, y_min, y_max)` | Create plot |
| `plot_series_line(id, x, y, r, g, b)` | Add line series |
| `plot_series_scatter(id, x, y, r, g, b)` | Add scatter series |
| `plot_series_bar(id, x, y, r, g, b, width)` | Add bar series |
| `plot_series_candle(id, x, o, h, l, c, ...)` | Add candlestick series |
| `plot_autoscale(id)` | Auto-fit axis ranges |
| `plot_grid(id, x_ticks, y_ticks)` | Draw grid lines |
| `plot_crosshair(id, mx, my)` | Draw crosshair |
| `plot_draw(id)` | Render all series |
| `plot_clear(id)` | Clear all series |

### `gui/theme.flow`

```
use gui/theme
```

| Function | Description |
|----------|-------------|
| `gui_theme_dark()` | Dark theme |
| `gui_theme_light()` | Light theme |
| `gui_theme_ocean()` | Ocean blue theme |

### `gui/buffer_view.flow`

```
use gui/buffer_view
```

Visualize data arrays on canvas.

| Function | Description |
|----------|-------------|
| `buffer_view_image(canvas, r, g, b, w, h)` | Display image |
| `buffer_view_heatmap(canvas, arr, w, h, min, max)` | Heatmap visualization |
| `buffer_view_waveform(canvas, arr, r, g, b)` | Waveform plot |
| `buffer_view_histogram(canvas, arr, bins, r, g, b)` | Histogram |

---

## Web

### `web/http.flow`

```
use web/http
```

| Function | Signature | Description |
|----------|-----------|-------------|
| `http_get_json` | `(url)` | GET and parse JSON |
| `http_post_json` | `(url, data)` | POST JSON and parse response |
| `api_call` | `(method, url, body, headers_map)` | Generic API call |

Note: Built-in `http_get`, `http_post`, `json_parse`, `json_stringify` available without import.

### `web/json_util.flow`

```
use web/json_util
```

| Function | Signature | Description |
|----------|-----------|-------------|
| `json_pretty` | `(data)` | Pretty-print JSON |
| `json_merge` | `(a, b)` | Merge two maps |
| `json_get` | `(data, path)` | Get nested value by path |
| `json_flatten` | `(data)` | Flatten nested map |

### `web/url.flow`

```
use web/url
```

| Function | Signature | Description |
|----------|-----------|-------------|
| `url_parse` | `(url)` | Parse URL to map (protocol, host, path, query) |
| `build_url` | `(base, path, params)` | Build URL from parts |
| `query_string` | `(params)` | Map to query string |

### `web/server.flow`

```
use web/server
```

HTTP server helpers (wraps builtins).

| Function | Signature | Description |
|----------|-----------|-------------|
| `http_serve` | `(port, handler)` | Start HTTP server |
| `text_response` | `(body)` | Plain text response |
| `json_response` | `(data)` | JSON response |
| `html_response` | `(html)` | HTML response |
| `redirect_response` | `(url)` | Redirect response |
| `error_response` | `(code, msg)` | Error response |
| `ok_json` | `(data)` | Shorthand JSON 200 |
| `ok_html` | `(html)` | Shorthand HTML 200 |
| `not_found` | `()` | 404 response |
| `method_not_allowed` | `()` | 405 response |

---

## DevOps

### `devops/config.flow`

```
use devops/config
```

| Function | Signature | Description |
|----------|-----------|-------------|
| `parse_ini` | `(content)` | Parse INI-style config |
| `parse_env_file` | `(content)` | Parse .env file |
| `config_get` | `(config, key, default)` | Get with default |

### `devops/fs.flow`

```
use devops/fs
```

| Function | Signature | Description |
|----------|-----------|-------------|
| `find_files` | `(dir, ext)` | Find files by extension |
| `walk_dir` | `(dir)` | Recursive directory listing |
| `file_info` | `(path)` | File metadata map |
| `glob_files` | `(dir, pattern)` | Glob pattern matching |
| `copy_file` | `(src, dst)` | Copy file |
| `path_join` | `(a, b)` | Join path components |
| `path_parent` | `(path)` | Parent directory |
| `path_name` | `(path)` | File name |
| `path_ext` | `(path)` | File extension |

### `devops/log.flow`

```
use devops/log
```

| Function | Signature | Description |
|----------|-----------|-------------|
| `log_set_level` | `(level)` | Set log level (0=DEBUG..3=ERROR) |
| `log_debug` | `(msg)` | Debug message |
| `log_info` | `(msg)` | Info message |
| `log_warn` | `(msg)` | Warning message |
| `log_error` | `(msg)` | Error message |
| `log_timed` | `(msg)` | Message with timestamp |

### `devops/process.flow`

```
use devops/process
```

| Function | Signature | Description |
|----------|-----------|-------------|
| `run` | `(cmd)` | Execute command, return output |
| `run_shell` | `(cmd)` | Execute via shell |
| `run_status` | `(cmd)` | Execute and return exit status |
| `run_ok` | `(cmd)` | 1.0 if command succeeds |
| `pipe` | `(cmd1, cmd2)` | Pipe output of cmd1 to cmd2 |
| `env_get` | `(name)` | Get environment variable |
| `which` | `(name)` | Find executable path |

### `devops/template.flow`

```
use devops/template
```

| Function | Signature | Description |
|----------|-----------|-------------|
| `render` | `(template, vars)` | Replace `{key}` with values |
| `load_template` | `(path)` | Load template from file |
| `render_file` | `(path, vars)` | Load and render template |

---

## System

### `sys/args.flow`

```
use sys/args
```

```flow
let args = parse_args()
let verbose = arg_flag(args, "--verbose")
let out = arg_value(args, "--output", "default.csv")
```

| Function | Signature | Description |
|----------|-----------|-------------|
| `parse_args` | `()` | Parse CLI arguments to map |
| `arg_flag` | `(args, name)` | 1.0 if flag present |
| `arg_value` | `(args, name, default)` | Get flag value or default |
| `arg_required` | `(args, name)` | Get required flag or panic |

### `sys/env.flow`

```
use sys/env
```

| Function | Signature | Description |
|----------|-----------|-------------|
| `get_env` | `(name)` | Get env var |
| `get_env_or` | `(name, default)` | Get env var with fallback |
| `get_home` | `()` | Home directory |
| `get_cwd` | `()` | Current working directory |
| `get_path` | `()` | PATH variable |
| `get_user` | `()` | Current username |
| `get_temp_dir` | `()` | Temp directory |

### `sys/platform.flow`

```
use sys/platform
```

| Function | Signature | Description |
|----------|-----------|-------------|
| `get_os` | `()` | OS name |
| `is_windows` | `()` | 1.0 on Windows |
| `is_linux` | `()` | 1.0 on Linux |
| `is_mac` | `()` | 1.0 on macOS |
| `home_dir` | `()` | Home directory |
| `temp_dir` | `()` | Temp directory |
| `user_name` | `()` | Current user |
| `path_separator` | `()` | OS path separator |

### `sys/memory.flow`

```
use sys/memory
```

| Function | Signature | Description |
|----------|-----------|-------------|
| `gpu_mem_total` | `()` | Total GPU memory |
| `gpu_mem_used` | `()` | Used GPU memory |
| `gpu_device_name` | `()` | GPU device name |
| `format_bytes` | `(n)` | Human-readable size |

### `sys/timer.flow`

```
use sys/timer
```

| Function | Signature | Description |
|----------|-----------|-------------|
| `stopwatch` | `()` | Start timer (returns ms timestamp) |
| `elapsed` | `(start)` | Milliseconds since start |
| `elapsed_secs` | `(start)` | Seconds since start |
| `benchmark` | `(label)` | Print benchmark start |
| `benchmark_end` | `(label, start)` | Print benchmark result |
| `sleep_busy` | `(ms)` | Busy-wait sleep |
| `format_duration` | `(ms)` | Human-readable duration |

### `sys/test.flow`

```
use sys/test
```

Test framework for `.flow` programs.

```flow
use sys/test
assert_eq(add(2, 3), 5.0, "add works")
assert_near(3.14, PI, 0.01, "pi approx")
print_summary()
```

| Function | Signature | Description |
|----------|-----------|-------------|
| `test_suite` | `(name)` | Declare test suite |
| `assert_eq` | `(got, expected, msg)` | Assert equality |
| `assert_near` | `(got, expected, tol, msg)` | Assert within tolerance |
| `assert_true` | `(val, msg)` | Assert truthy |
| `assert_false` | `(val, msg)` | Assert falsy |
| `assert_str_eq` | `(got, expected, msg)` | Assert string equality |
| `assert_gt` | `(a, b, msg)` | Assert a > b |
| `assert_lt` | `(a, b, msg)` | Assert a < b |
| `print_summary` | `()` | Print pass/fail summary |

---

## Time

### `time/datetime.flow`

```
use time/datetime
```

| Function | Signature | Description |
|----------|-----------|-------------|
| `unix_now` | `()` | Current Unix timestamp |
| `ms_now` | `()` | Current time in milliseconds |
| `unix_to_date` | `(ts)` | Timestamp to date map |
| `is_leap_year` | `(year)` | 1.0 if leap year |
| `format_date` | `(ts)` | Format as YYYY-MM-DD |
| `format_datetime_full` | `(ts)` | Full datetime string |
| `str_zfill` | `(s, width)` | Zero-pad string |
| `day_of_week` | `(ts)` | Day of week (0=Sunday) |
| `day_name` | `(ts)` | Day name string |
| `days_between` | `(ts1, ts2)` | Days between timestamps |
| `hours_between` | `(ts1, ts2)` | Hours between timestamps |

---

## Game

### `game/ecs.flow`

```
use game/ecs
```

Data-oriented Entity Component System using parallel arrays.

| Function | Signature | Description |
|----------|-----------|-------------|
| `ecs_new` | `()` | Create ECS world |
| `ecs_create` | `(ecs)` | Create entity, return ID |
| `ecs_find` | `(ecs, name)` | Find entity by name |
| `ecs_add_pos` | `(ecs, id)` | Add position component |
| `ecs_add_pos_at` | `(ecs, id, x, y)` | Add position at (x, y) |
| `ecs_add_vel` | `(ecs, id, vx, vy)` | Add velocity component |
| `ecs_add_hp` | `(ecs, id, hp)` | Add health component |
| `ecs_update_physics` | `(ecs, dt)` | Update positions by velocity |
| `ecs_damage` | `(ecs, id, amount)` | Apply damage |
| `ecs_get_pos` | `(ecs, id)` | Get position map |
| `ecs_alive_count` | `(ecs)` | Count entities with HP > 0 |

### `game/physics2d.flow`

```
use game/physics2d
```

2D physics simulation using parallel arrays (SoA layout).

| Function | Signature | Description |
|----------|-----------|-------------|
| `phys_add_body` | `(x, y, vx, vy, mass, radius)` | Add rigid body |
| `phys_gravity` | `(ax, ay)` | Apply gravity acceleration |
| `phys_integrate` | `(dt)` | Euler integration step |
| `phys_bounce_walls` | `(w, h, restitution)` | Wall collision response |
| `phys_dist_sq` | `(i, j)` | Squared distance between bodies |
| `phys_collide_pairs` | `(restitution)` | Pairwise collision |
| `phys_kinetic_energy` | `()` | Total kinetic energy |

---

## Algorithms

### `algo/pathfind.flow`

```
use algo/pathfind
```

Grid-based pathfinding. Grid is a flat array: `grid[y * width + x]`, 0.0 = passable, 1.0 = wall.

| Function | Signature | Description |
|----------|-----------|-------------|
| `manhattan` | `(x1, y1, x2, y2)` | Manhattan distance |
| `astar` | `(grid, w, h, sx, sy, gx, gy)` | A* pathfinding |

Returns path as `[x1, y1, x2, y2, ...]` from start to goal.

---

## Formats

### `formats/gguf.flow`

```
use formats/gguf
```

GGUF model file parser for LLM weight loading.

| Function | Description |
|----------|-------------|
| `gguf_load(path)` | Load and parse GGUF file |
| `gguf_load_from_bytes(bytes)` | Parse GGUF from byte array |

Returns model map with metadata (`kv.*`) and tensor info (`t.*.type/offset/count`).

---

## LLM / AI

The `llm/` and `ai/` directories contain the full LLM inference stack, written in pure `.flow`.

### `llm/tokenizer.flow` / `ai/tokenizer.flow`

```
use llm/tokenizer
```

| Function | Description |
|----------|-------------|
| `tokenizer_load(vocab_file)` | Load vocabulary from JSON |
| `tokenizer_encode(text, vocab)` | Encode text to token IDs |
| `tokenizer_decode(ids, vocab)` | Decode token IDs to text |

### `llm/sampling.flow` / `ai/sampling.flow`

```
use llm/sampling
```

| Function | Description |
|----------|-------------|
| `sample_greedy(logits)` | Argmax sampling |
| `sample_top_k(logits, k, temp)` | Top-k sampling |
| `sample_top_p(logits, p, temp)` | Nucleus (top-p) sampling |
| `sample_temperature(logits, temp)` | Temperature sampling |
| `sample_min_p(logits, min_p, temp)` | Min-p sampling |

### `llm/embed.flow` / `ai/embed.flow`

```
use llm/embed
```

| Function | Description |
|----------|-------------|
| `embed_token(model_path, model, token_id)` | Load one embedding row |
| `embed_batch(model_path, model, token_ids, n_embd)` | Batch embed |

### `llm/weight_loader.flow` / `ai/weight_loader.flow`

```
use llm/weight_loader
```

| Function | Description |
|----------|-------------|
| `load_layer_weights(file_bytes, model, layer_idx)` | Load transformer layer weights |
| `load_embedding_row(file_bytes, model, token_id, n_embd)` | Load embedding |
| `load_output_norm(file_bytes, model)` | Load output normalization weights |
| `load_output_proj(file_bytes, model)` | Load output projection |
| `read_tensor_by_name(file_bytes, model, tname)` | Read arbitrary tensor |

### `llm/ops.flow`

```
use llm/ops
```

CPU implementations of transformer operations.

| Function | Description |
|----------|-------------|
| `rmsnorm_cpu(hidden, weight, n, eps)` | RMS normalization |
| `silu_cpu(arr, n)` | SiLU activation |
| `gqa_expand(v, n_head, n_kv_head, head_dim)` | GQA expansion |
| `vec_add(a, b, n)` | Element-wise addition |
| `rope_cpu(vec, pos, head_dim, n_heads, theta)` | Rotary position embedding |

---

## Loom Engine

The `loom/` directory contains the GPU compute engine — SPIR-V kernel emitters, GPU dispatch, neural network kernels, and high-level GPU operations. See [loom-engine.md](loom-engine.md) for details.

Key subdirectories:
- `loom/ops/` — GPU dispatch, runtime, patterns, homeostasis
- `loom/nn/` — Attention, FFN, RMSNorm, RoPE, SiLU, softmax, dequantization
- `loom/math/` — GPU linear algebra, statistics, signal processing
- `loom/data/` — GPU array ops, aggregation, composite operations
- `loom/emit/` — SPIR-V kernel code generators

---

## Self-Hosted Compiler

The `compiler/` directory contains the self-hosted OctoFlow compiler (24,000+ lines of `.flow`). See [annex-n-self-hosting.md](annex-n-self-hosting.md) for details.

Key modules:
- `compiler/lexer.flow` — Tokenizer
- `compiler/parser.flow` — Recursive-descent parser
- `compiler/ir.flow` — Intermediate representation
- `compiler/codegen.flow` — Code generation
- `compiler/eval.flow` — AST evaluator
- `compiler/preflight.flow` — Pre-flight validation
- `compiler/spirv_emit.flow` — SPIR-V code generation

---

## Quick Reference: Module Count by Domain

| Domain | Modules | Functions |
|--------|---------|-----------|
| Core (math, sort, array) | 3 | 28 |
| Collections | 5 | 40+ |
| String | 3 | 25 |
| Data | 6 | 35+ |
| Database | 3 | 15 |
| Statistics | 7 | 75+ |
| Machine Learning | 9 | 65+ |
| Science | 7 | 50+ |
| Crypto | 3 | 15 |
| Encoding | 2 | 8 |
| Media | 10 | 40+ |
| Terminal | 5 | 6 |
| GUI | 8 | 90+ |
| Web | 4 | 20+ |
| DevOps | 5 | 25 |
| System | 6 | 35 |
| Time | 1 | 11 |
| Game | 2 | 18 |
| Algorithms | 1 | 2 |
| Formats | 1 | 2 |
| LLM/AI | 15+ | 30+ |
| Loom Engine | 40+ | 100+ |
| Compiler | 30+ | 200+ |

**Total: 423 modules, 945+ functions**
