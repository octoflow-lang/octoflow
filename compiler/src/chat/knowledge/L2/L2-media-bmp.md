# bmp (L2)
media/bmp — BMP image decoder (24-bit BGR, 32-bit BGRA, 8-bit palette)

## Functions
bmp_parse(data: array) → map
  Parse BMP header, return metadata (width, height, bpp)
bmp_decode(data: array) → map
  Decode BMP to r/g/b pixel arrays and dimensions
