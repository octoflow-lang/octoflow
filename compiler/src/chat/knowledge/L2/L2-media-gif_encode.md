# gif_encode (L2)
media/gif_encode — GIF encoder (LZW compression, 256-color palette, animated GIF89a)

## Functions
gif_encode(r: array, g: array, b: array, w: int, h: int) → array
  Encode single-frame GIF from RGB arrays
gif_encode_frames(frames: array, w: int, h: int, delay: int) → array
  Encode animated GIF from array of frames with delay (centiseconds)
