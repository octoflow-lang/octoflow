# OctoFlow Standard Library — All Signatures

Import with `use domain.module` before calling functions.

---

## collections.collections
```
use collections.collections
stack_new() → map                    stack_push(s, val)
stack_pop(s) → value                 stack_peek(s) → value
stack_is_empty(s) → f32              queue_new() → map
queue_enqueue(q, val)                queue_dequeue(q) → value
queue_size(q) → f32                  queue_is_empty(q) → f32
counter_new() → map                  counter_add(c, key)
counter_get(c, key) → f32            counter_from_array(arr) → map
set_new() → map                      set_add(s, val)
set_has(s, val) → f32                set_remove(s, val)
set_size(s) → f32                    dmap_new(default) → map
dmap_get(m, key) → value             dmap_set(m, key, val)
pair(a, b) → map                     pair_first(p) → value
pair_second(p) → value
```

## collections.heap
```
use collections.heap
heap_create() → map                  heap_push(h, val)
heap_pop(h) → value                  heap_peek(h) → value
heap_size(h) → f32                   heap_is_empty(h) → f32
heapify(arr) → map
```

## collections.graph
```
use collections.graph
graph_create() → map                 graph_add_node(g, name)
graph_add_edge(g, from, to, weight)  graph_neighbors(g, node) → [string]
graph_weight(g, from, to) → f32      graph_nodes(g) → [string]
graph_bfs(g, start) → [string]       graph_has_edge(g, from, to) → f32
```

## collections.stack
```
use collections.stack
stack_create() → map                 stack_push(s, val)
stack_pop(s) → value                 stack_peek(s) → value
stack_size(s) → f32                  stack_is_empty(s) → f32
stack_to_array(s) → [value]
```

## collections.queue
```
use collections.queue
queue_create() → map                 queue_enqueue(q, val)
queue_dequeue(q) → value             queue_peek(q) → value
queue_size(q) → f32                  queue_is_empty(q) → f32
queue_to_array(q) → [value]
```

---

## stats.descriptive
```
use stats.descriptive
skewness(arr) → f32                  kurtosis(arr) → f32
iqr(arr) → f32                       percentile(arr, p) → f32
weighted_mean(values, weights) → f32 trimmed_mean(arr, pct) → f32
describe(arr) → map                  mode(arr) → f32
geometric_mean(arr) → f32            harmonic_mean(arr) → f32
coeff_of_variation(arr) → f32        zscore(arr) → [f32]
```

## stats.hypothesis
```
use stats.hypothesis
t_test_one_sample(arr, mu) → map     t_test_two_sample(a, b) → map
paired_t_test(a, b) → map            chi_squared(obs, exp) → map
z_test(arr, mu, sigma) → map
```

## stats.correlation
```
use stats.correlation
pearson(x, y) → f32                  spearman(x, y) → f32
rank_array(arr) → [f32]             covariance(x, y) → f32
linear_fit(x, y) → map              residuals(x, y) → [f32]
polynomial_fit(x, y, degree) → [f32]
```

## stats.distribution
```
use stats.distribution
normal_pdf(x, mu, sigma) → f32      normal_cdf(x, mu, sigma) → f32
normal_inv(p, mu, sigma) → f32      uniform_random(lo, hi) → f32
normal_random(mu, sigma) → f32      exponential_random(lambda) → f32
poisson_pmf(k, lambda) → f32        binomial_pmf(k, n, p) → f32
combinations(n, k) → f32            random_sample(arr, n) → [value]
```

## stats.timeseries
```
use stats.timeseries
sma(arr, period) → [f32]            ema(arr, period) → [f32]
wma(arr, period) → [f32]            rsi(arr, period) → [f32]
macd(arr, fast, slow) → [f32]       macd_signal(arr, fast, slow, sig) → [f32]
macd_histogram(arr, fast, slow, sig) → [f32]
bollinger(arr, period, mult) → [f32] bollinger_upper(arr, p, m) → [f32]
bollinger_lower(arr, p, m) → [f32]  atr(high, low, close, period) → [f32]
vwap(close, volume) → [f32]         returns(arr) → [f32]
log_returns(arr) → [f32]            drawdown(arr) → [f32]
rolling(arr, period, fn_name) → [f32] diff(arr) → [f32]
cumsum(arr) → [f32]                 autocorrelation(arr, lag) → f32
```

## stats.risk
```
use stats.risk
sharpe_ratio(returns, rf) → f32     sortino_ratio(returns, rf) → f32
max_drawdown(equity) → f32          calmar_ratio(returns, equity) → f32
value_at_risk(returns, conf) → f32  expected_shortfall(returns, conf) → f32
volatility(returns) → f32           beta(asset, market) → f32
alpha(asset, market, rf) → f32      information_ratio(active, bench) → f32
win_rate(trades) → f32              profit_factor(trades) → f32
expectancy(trades) → f32            covariance_xy(x, y) → f32
```

## stats.math_ext
```
use stats.math_ext
factorial(n) → f32                   permutations(n, r) → f32
gcd(a, b) → f32                     lcm(a, b) → f32
is_prime(n) → f32                    fibonacci(n) → f32
power_mod(base, exp, mod) → f32     sigmoid(x) → f32
tanh_fn(x) → f32                    relu(x) → f32
softplus(x) → f32                   logistic(x, L, k, x0) → f32
linspace(start, end, n) → [f32]     arange(start, end, step) → [f32]
dot_product(a, b) → f32             magnitude(v) → f32
normalize_vec(v) → [f32]            cross_product_3d(a, b) → [f32]
```

---

## ml.classify
```
use ml.classify
knn_predict(train_x, train_y, test, k) → f32
logistic_regression(x, y, lr, epochs) → [f32]
logistic_predict(x, weights) → f32
naive_bayes_train(x, y) → map
naive_bayes_predict(model, x) → f32
```

## ml.regression
```
use ml.regression
linear_regression(x, y) → map       ridge_regression(x, y, alpha) → map
predict_linear(model, x) → f32      r_squared(actual, predicted) → f32
gradient_descent_linear(x, y, lr, epochs) → map
```

## ml.linalg
```
use ml.linalg
mat_create(rows, cols, fill) → map   mat_identity(n) → map
mat_get(m, r, c) → f32              mat_set(m, r, c, val)
mat_mul(a, b) → map                  mat_transpose(m) → map
mat_add(a, b) → map                  mat_scale(m, s) → map
mat_determinant(m) → f32            mat_inverse(m) → map
mat_trace(m) → f32                   mat_solve(A, b) → [f32]
```

## ml.metrics
```
use ml.metrics
accuracy(actual, predicted) → f32    precision(actual, predicted) → f32
recall(actual, predicted) → f32      f1_score(actual, predicted) → f32
mse(actual, predicted) → f32         rmse(actual, predicted) → f32
mae(actual, predicted) → f32         confusion_matrix(actual, pred) → map
mean_absolute_percentage_error(a, p) → f32
```

## ml.nn
```
use ml.nn
dense_forward(input, weights, bias) → [f32]
relu_forward(x) → [f32]             sigmoid_forward(x) → [f32]
tanh_forward(x) → [f32]             softmax(x) → [f32]
cross_entropy_loss(pred, target) → f32
mse_loss(pred, target) → f32        init_weights(rows, cols) → [f32]
init_bias(size) → [f32]             sgd_update(params, grads, lr) → [f32]
dropout(x, rate) → [f32]            batch_norm(x) → [f32]
```

## ml.cluster
```
use ml.cluster
kmeans(data, k, dim, iters) → map
euclidean_distance(a, b) → f32       manhattan_distance(a, b) → f32
silhouette_score(data, labels, k, dim) → f32
```

## ml.tree
```
use ml.tree
gini_impurity(labels) → f32         decision_stump(x, y) → map
find_best_split(data, labels, n_features) → map
```

## ml.ensemble
```
use ml.ensemble
bagging_predict(models, x) → f32    weighted_vote(preds, weights) → f32
bootstrap_sample(arr) → [value]     bootstrap_indices(n) → [f32]
```

## ml.preprocess
```
use ml.preprocess
train_test_split(x, y, ratio) → map shuffle_array(arr) → [value]
minmax_scale(arr) → [f32]           zscore_scale(arr) → [f32]
feature_scale(arr, method) → [f32]  encode_labels(arr) → map
impute_missing(arr, strategy) → [f32]
```

---

## science.calculus
```
use science.calculus
derivative(arr, dx) → [f32]         integrate_trapz(arr, dx) → f32
cumulative_trapz(arr, dx) → [f32]   second_derivative(arr, dx) → [f32]
```

## science.signal
```
use science.signal
convolve(signal, kernel) → [f32]    moving_avg_filter(arr, w) → [f32]
gaussian_kernel(size, sigma) → [f32] hamming_window(n) → [f32]
hanning_window(n) → [f32]           blackman_window(n) → [f32]
cross_correlate(a, b) → [f32]       bandpass(arr, lo, hi, sr) → [f32]
envelope(arr) → [f32]               zero_crossings(arr) → [f32]
peak_detect(arr, threshold) → [f32]
```

## science.interpolate
```
use science.interpolate
linear_interp(x0, y0, x1, y1, x) → f32
linear_interp_array(xs, ys, x_new) → [f32]
bilinear_interp(q11, q12, q21, q22, x, y) → f32
nearest_interp(xs, ys, x) → f32
```

## science.optimize
```
use science.optimize
gradient_descent(fn_name, x0, lr, iters) → f32
golden_section(fn_name, a, b, tol) → f32
newton_raphson(fn_name, dfn_name, x0, tol, max) → f32
bisection(fn_name, a, b, tol) → f32
integrate_trapezoid(fn_name, a, b, n) → f32
integrate_simpson(fn_name, a, b, n) → f32
differentiate(fn_name, x, h) → f32
```

## science.physics
```
use science.physics
integrate_euler(fn_name, y0, t0, t1, dt) → [f32]
integrate_rk4(fn_name, y0, t0, t1, dt) → [f32]
spring_damper(k, c, m, x0, v0, dt, steps) → [f32]
projectile(v0, angle, dt) → map
kinetic_energy(m, v) → f32          potential_energy(m, h) → f32
gravitational_force(m1, m2, r) → f32
wave_equation_1d(u, c, dx, dt, steps) → [f32]
```

---

## data.csv
```
use data.csv
csv_read(path) → [map]              csv_write(path, rows)
csv_filter(rows, col, op, val) → [map]
csv_sort(rows, col, asc) → [map]    csv_select(rows, cols) → [map]
csv_group(rows, col) → map          csv_count(rows) → f32
csv_unique(rows, col) → [value]     csv_column(rows, col) → [value]
csv_add_column(rows, col, fn_name) → [map]
```

## data.transform
```
use data.transform
normalize(arr) → [f32]              standardize(arr) → [f32]
one_hot(labels, n_classes) → [f32]  clip(arr, lo, hi) → [f32]
interpolate_missing(arr) → [f32]    resample(arr, new_len) → [f32]
bin_data(arr, n_bins) → [f32]       scale_to_range(arr, lo, hi) → [f32]
```

## data.validate
```
use data.validate
is_numeric(s) → f32                  is_email(s) → f32
is_url(s) → f32                     in_range(x, lo, hi) → f32
clamp_value(x, lo, hi) → f32
```

## data.io
```
use data.io
load_json(path) → map               save_json(path, data)
load_text(path) → string            save_text(path, text)
load_lines(path) → [string]         file_size_str(path) → string
file_exists(path) → f32
```

## data.pipeline
```
use data.pipeline
chain_apply(arr, fns) → [value]     batch_process(arr, batch_size, fn_name) → [value]
parallel_merge(arr1, arr2, fn_name) → [value]
```

---

## string.string
```
use string.string
str_center(s, width, fill) → string str_ljust(s, width, fill) → string
str_rjust(s, width, fill) → string  str_count(s, sub) → f32
str_repeat(s, n) → string           str_reverse(s) → string
str_is_digit(s) → f32               str_is_alpha(s) → f32
str_is_empty(s) → f32               str_lstrip(s) → string
str_rstrip(s) → string              str_title(s) → string
str_zfill(s, width) → string        str_join(arr, sep) → string
str_split(s, sep) → [string]
```

## string.format
```
use string.format
pad_left(s, width, ch) → string     pad_right(s, width, ch) → string
format_number(n, decimals) → string format_bytes(n) → string
format_duration(seconds) → string
```

## string.regex
```
use string.regex
glob_match(pattern, text) → f32     str_escape(s) → string
word_count(s) → f32
```

---

## crypto.hash
```
use crypto.hash
djb2(s) → f32                       fnv1a(s) → f32
checksum(s) → f32                   crc_simple(s) → f32
```

## crypto.encoding
```
use crypto.encoding
to_base64(s) → string               from_base64(s) → string
to_hex(s) → string                  from_hex(s) → string
```

## crypto.random
```
use crypto.random
random_float(lo, hi) → f32          random_int(lo, hi) → f32
random_hex(n) → string              uuid_v4() → string
random_token(n) → string            random_choice(arr) → value
random_shuffle(arr) → [value]
```

---

## web.http
```
use web.http
http_get_json(url) → map            http_post_json(url, data) → map
api_call(method, url, body) → map
```

## web.json_util
```
use web.json_util
json_pretty(obj) → string           json_merge(a, b) → map
json_get(obj, path) → value         json_flatten(obj) → map
```

## web.url
```
use web.url
url_parse(url) → map                build_url(base, params) → string
query_string(params) → string
```

---

## devops.fs
```
use devops.fs
find_files(dir, pattern) → [string] walk_dir(dir) → [string]
file_info(path) → map               glob_files(pattern) → [string]
copy_file(src, dst)                  path_join(a, b) → string
path_parent(p) → string             path_name(p) → string
path_ext(p) → string
```

## devops.process
```
use devops.process
run(cmd, args) → map                 run_shell(cmd) → map
run_status(cmd, args) → f32          run_ok(cmd, args) → f32
pipe(cmd1, cmd2) → map              env_get(name) → string
which(name) → string
```

## devops.log
```
use devops.log
log_set_level(level)                 log_debug(msg)
log_info(msg)                        log_warn(msg)
log_error(msg)                       log_timed(label)
```

## devops.config
```
use devops.config
parse_ini(path) → map               parse_env_file(path) → map
config_get(cfg, section, key) → value
```

## devops.template
```
use devops.template
render(template, vars) → string     load_template(path) → string
render_file(path, vars) → string
```

---

## db.core
```
use db.core
db_create() → map                    db_table(db, name)
db_tables(db) → [string]            db_insert(db, table, row)
db_count(db, table) → f32           db_select_column(db, table, col) → [value]
db_select_row(db, table, idx) → map db_where(db, table, col, op, val) → [map]
db_import_csv(db, table, path)
```

## db.query
```
use db.query
db_join(db, t1, t2, key) → [map]   db_group_by(db, table, col) → map
db_order_by(db, table, col, asc) → [map]
```

## db.schema
```
use db.schema
db_columns(db, table) → [string]    db_describe(db, table) → map
db_rename_column(db, table, old, new)
```

---

## sys.args
```
use sys.args
parse_args() → map                   arg_flag(args, name) → f32
arg_value(args, name) → string      arg_required(args, name) → string
```

## sys.env
```
use sys.env
get_env(name) → string              get_env_or(name, default) → string
get_home() → string                 get_cwd() → string
get_path() → string                 get_user() → string
get_temp_dir() → string
```

## sys.timer
```
use sys.timer
stopwatch() → f32                    elapsed(start) → f32
elapsed_secs(start) → f32           benchmark(label) → f32
benchmark_end(start, label)          sleep_busy(ms)
format_duration(ms) → string
```

## sys.platform
```
use sys.platform
get_os() → string                    is_windows() → f32
is_linux() → f32                     is_mac() → f32
home_dir() → string                 temp_dir() → string
user_name() → string                path_separator() → string
```

## sys.memory
```
use sys.memory
gpu_mem_total() → f32               gpu_mem_used() → f32
gpu_device_name() → string          format_bytes(n) → string
```

## sys.test
```
use sys.test
test_suite(name) → map              assert_eq(suite, actual, expected, msg)
assert_near(suite, actual, expected, tol, msg)
assert_true(suite, cond, msg)       assert_false(suite, cond, msg)
assert_str_eq(suite, a, b, msg)     assert_gt(suite, a, b, msg)
assert_lt(suite, a, b, msg)         print_summary(suite)
```

---

## time.datetime
```
use time.datetime
unix_now() → f32                     ms_now() → f32
unix_to_date(ts) → string           is_leap_year(y) → f32
format_date(ts) → string            format_datetime_full(ts) → string
day_of_week(ts) → f32               day_name(ts) → string
days_between(ts1, ts2) → f32        hours_between(ts1, ts2) → f32
```

---

## Core Utility Modules (no domain prefix)

### math.flow
```
tap("stdlib/math.flow")
min_of(a, b) → f32                  max_of(a, b) → f32
clamp_val(x, lo, hi) → f32          lerp(a, b, t) → f32
map_range(x, in_lo, in_hi, out_lo, out_hi) → f32
sign(x) → f32                       deg_to_rad(d) → f32
rad_to_deg(r) → f32
```

### sort.flow
```
tap("stdlib/sort.flow")
insertion_sort(arr) → [value]        bubble_sort(arr) → [value]
```

### array_utils.flow
```
tap("stdlib/array_utils.flow")
arr_contains(arr, val) → f32         arr_sum(arr) → f32
arr_avg(arr) → f32                  arr_min(arr) → f32
arr_max(arr) → f32                  swap(arr, i, j)
fill(n, val) → [value]             binary_search(arr, target) → f32
```

### io.flow
```
tap("stdlib/io.flow")
io_read_file(path) → string         io_write_file(path, content)
io_append_file(path, content)        io_read_bytes(path) → [f32]
io_write_bytes(path, bytes)          io_puts(text)
io_remove(path) → f32               io_file_size(path) → f32
```
