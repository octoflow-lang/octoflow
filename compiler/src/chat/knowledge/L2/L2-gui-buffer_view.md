# buffer_view (L2)
gui/buffer_view — GPU array visualization (image, heatmap, waveform, histogram)

## Functions
buffer_view_image(canvas_id: int, r: array, g: array, b: array, w: int, h: int) → float
  Display RGB arrays as image on canvas
buffer_view_heatmap(canvas_id: int, data: array, w: int, h: int) → float
  Display 2D array as color heatmap
buffer_view_waveform(canvas_id: int, data: array, color: int) → float
  Display 1D array as waveform plot
buffer_view_histogram(canvas_id: int, data: array, bins: int, color: int) → float
  Display data distribution as histogram
