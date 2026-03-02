# plot_create (L3)

## Working Example
```flow
use gui/plot

let chart = plot_create("Temperature Over Time")

let mut xs = []
let mut ys = []
for i in range(0, 24)
  push(xs, i)
  let temp = 18.0 + 7.0 * sin(i * 0.26)
  push(ys, temp)
end

plot_series_line(chart, "sensor-A", xs, ys)

let mut ys2 = []
for i in range(0, 24)
  let temp = 20.0 + 5.0 * sin(i * 0.26 + 1.0)
  push(ys2, temp)
end

plot_series_line(chart, "sensor-B", xs, ys2)

plot_autoscale(chart)
plot_draw(chart)

let points = len(xs)
print("plotted 2 series with {points} points each")
```

## Expected Output
```
plotted 2 series with 24 points each
```

*(A line chart window opens showing two sinusoidal temperature curves.)*

## Common Mistakes
- DON'T: `plot_create()` with no title → DO: `plot_create("My Chart")`
- DON'T: `chart.add_series(...)` → DO: `plot_series_line(chart, ...)`
- DON'T: `push(xs, i);` → DO: `push(xs, i)` (no semicolons)

## Edge Cases
- plot_autoscale adjusts axes to fit all data; call before plot_draw
- plot_series_line with mismatched xs/ys lengths uses the shorter length
- plot_draw blocks until the chart window is closed
