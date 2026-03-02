# gif (L2)
media/gif — GIF decoder (LZW decompression, color tables, multi-frame animation)

## Functions
gif_decode(data: array) → map
  Decode GIF, return frames with palette and dimensions
gif_extract_r(frame: map) → array
  Extract red channel as pixel array
gif_extract_g(frame: map) → array
  Extract green channel as pixel array
gif_extract_b(frame: map) → array
  Extract blue channel as pixel array
