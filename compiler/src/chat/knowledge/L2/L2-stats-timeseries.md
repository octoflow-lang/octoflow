# timeseries (L2)
stats/timeseries — Time series and trading indicators

## Functions
sma | ema | wma(data: array, period: int) → array — Moving averages
rsi(data: array, period: int) → array — Relative strength index
macd(data: array, fast: int, slow: int) → array — MACD line
macd_signal(macd: array, period: int) → array | macd_histogram(macd: array, signal: array) → array
bollinger(data: array, period: int, n_std: float) → map — Bands (mid/upper/lower)
bollinger_upper | bollinger_lower(data: array, period: int, n_std: float) → array
atr(high: array, low: array, close: array, period: int) → array — Avg true range
vwap(price: array, volume: array) → array — Volume-weighted avg price
returns | log_returns | drawdown | cumsum(data: array) → array — Transforms
rolling(data: array, period: int, fn: string) → array — Rolling window
diff(data: array, lag: int) → array | autocorrelation(data: array, lag: int) → float
