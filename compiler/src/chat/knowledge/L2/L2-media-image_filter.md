# image_filter (L2)
media/image_filter — Convolution filters (blur, sharpen, edge detect, threshold, invert)

## Functions
img_convolve(pixels: array, w: int, h: int, kernel: array) → array
  Apply convolution kernel to pixel array
img_blur_box(pixels: array, w: int, h: int, radius: int) → array
  Box blur with given radius
img_sharpen(pixels: array, w: int, h: int) → array
  Sharpen using 3x3 kernel
img_edge_detect(pixels: array, w: int, h: int) → array
  Sobel edge detection
img_threshold(pixels: array, t: float) → array
  Binary threshold (pixels above t become 255)
img_invert(pixels: array) → array
  Invert pixel values (255 - v)
