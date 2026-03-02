# ttf (L2)
media/ttf — TrueType font parser (table directory, cmap, glyf, Bezier for SDF rendering)

## Functions
ttf_parse_tables(data: array) → map
  Parse font table directory (head, cmap, glyf, loca, etc.)
ttf_parse_all_ascii(data: array) → map
  Parse all printable ASCII glyphs to Bezier segment arrays
