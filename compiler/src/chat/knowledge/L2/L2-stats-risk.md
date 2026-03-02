# risk (L2)
stats/risk — Financial risk metrics

## Functions
sharpe_ratio | sortino_ratio(returns: array, rf: float) → float — Risk-adjusted return
max_drawdown(returns: array) → float — Largest peak-to-trough decline
calmar_ratio(returns: array, rf: float) → float — Return / max drawdown
value_at_risk | expected_shortfall(returns: array, confidence: float) → float — VaR/CVaR
volatility(returns: array) → float — Annualized std dev
beta(asset: array, market: array) → float — Market beta
alpha(asset: array, market: array, rf: float) → float — Jensen's alpha
information_ratio(returns: array, benchmark: array) → float — Excess/tracking error
win_rate(returns: array) → float | profit_factor(returns: array) → float
