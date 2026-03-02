# plot (L2)
gui/plot — Charting (line, scatter, bar, candlestick, autoscale, grid)

## Functions
plot_create(x, y, w, h: int) → int — create plot widget
plot_set_range(id: int, x_min, x_max, y_min, y_max: float) → float
plot_series_line(id: int, data: array, color: int) → float
plot_series_scatter(id: int, data: array, color: int) → float
plot_series_bar(id: int, data: array, color: int) → float
plot_series_candle(id: int, data: array, color: int) → float — OHLC
plot_autoscale(id: int) → float — fit axes to data
plot_grid(id, on: int) → float
plot_crosshair(id, on: int) → float
plot_draw(id: int) → float — render to canvas
