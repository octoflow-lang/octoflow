# gif_encode_gpu (L2)
media/gif_encode_gpu — GPU-accelerated GIF encoder (10-100x faster palette quantization)

## Functions
gif_rgb_to_indices_gpu(r: array, g: array, b: array, palette: array) → array
  GPU palette quantization, map RGB pixels to palette indices
gif_encode_frames_gpu(frames: array, w: int, h: int, delay: int) → array
  GPU-accelerated animated GIF encoding
