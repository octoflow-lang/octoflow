# image_util (L2)
media/image_util — Pixel access, geometry, color adjust, quality metrics

## Functions
img_get_pixel(px: array, w, x, y: int) → int
img_set_pixel(px: array, w, x, y, v: int) → float
img_fill(w, h, v: int) → array
img_copy(px: array) → array
img_crop(px: array, w, x, y, cw, ch: int) → array
img_paste(dst: array, dw: int, src: array, sw, x, y: int) → float
img_flip_h(px: array, w, h: int) → array
img_flip_v(px: array, w, h: int) → array
img_grayscale(r, g, b: array) → array
img_brightness(px: array, delta: float) → array
img_contrast(px: array, factor: float) → array
img_mse(a, b: array) → float — mean squared error
img_psnr(a, b: array) → float — peak signal-to-noise ratio
